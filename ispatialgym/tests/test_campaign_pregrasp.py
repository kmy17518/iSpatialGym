from types import SimpleNamespace as NS
import pytest
from ispatialgym.oracle import execute as E
from ispatialgym.oracle.execute import SkillError
from ispatialgym.sampling import campaign_runtime as C
from scripts.campaign_20261110.contracts import ContractError


def bind(monkeypatch,enabled=True):
    template=NS(class_id='inter_object_relation_satisfaction-ontop_small_scale_01',
      task='inter_object_relation_satisfaction-ontop_small_scale',
      scene_setup={'recovery_options':{'kielbasa_cartesian_pregrasp':enabled}})
    monkeypatch.setenv('ISPATIALGYM_CAMPAIGN_CONFIG','/unused')
    monkeypatch.setattr(C,'configuration',lambda:(None,template,None,None))
    return template


def consume(gen):
    actions=[]
    while True:
        try:actions.append(next(gen))
        except StopIteration as done:return actions,done.value


def test_complete_checked_transit_keeps_world_and_approach(monkeypatch):
    bind(monkeypatch);calls=[]
    def transit(*args,**kwargs):
        calls.append((args,kwargs));yield 'checked action'
    ctx={};targets={'pre_grasp':'pre','approach':['approach','grasp']}
    actions,done=consume(E.campaign_pregrasp_transit(NS(kind='library'),NS(cartesian_transit=transit),targets,ctx,'left',['support'],[]))
    assert actions==['checked action'] and done is True
    assert calls[0][1]=={'approach':['approach','grasp'],'ignore_objects':()}
    assert ctx['pre_grasp_cartesian']['ran'] is True


@pytest.mark.parametrize('moved',[False,True])
def test_fallback_is_only_before_any_action(monkeypatch,moved):
    bind(monkeypatch)
    def transit(*args,**kwargs):
        if moved:yield 'action'
        raise SkillError('plan_failed','no checked path',stage='cartesian_transit')
    gen=E.campaign_pregrasp_transit(NS(kind='library'),NS(cartesian_transit=transit),{'pre_grasp':'pre'},{},'left',['support'],[])
    if moved:
        with pytest.raises(SkillError):consume(gen)
    else:assert consume(gen)==([],False)


def test_unconfigured_handle_and_press_paths_are_unchanged(monkeypatch):
    t=bind(monkeypatch,False)
    assert consume(E.campaign_pregrasp_transit(NS(kind='library'),NS(),{}, {},'left',['support'],[]))==([],False)
    t.scene_setup['recovery_options']['kielbasa_cartesian_pregrasp']=True
    for kind,ignore in [('handle',[]),('library',['pressed_link'])]:
        assert consume(E.campaign_pregrasp_transit(NS(kind=kind),NS(),{}, {},'left',['support'],ignore))==([],False)
    t.class_id='different_class'
    with pytest.raises(ContractError):C.contact_recovery('kielbasa_cartesian_pregrasp')
