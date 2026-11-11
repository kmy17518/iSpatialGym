from dataclasses import asdict, replace
from unittest.mock import patch
import copy
import pytest
from scripts.campaign_20261110.contracts import ObjectRef, Template, digest, ContractError
from scripts.campaign_20261110 import materialize_training_table as m


def fixture():
    target=ObjectRef('target','target','beer_glass','glass',digest('glass'),default_scale=(.5,.5,.5),ready=True,readiness_source='catalog')
    support=ObjectRef('support','support','booth','table',digest('table'),default_scale=(2,2,1.5),ready=True,readiness_source='catalog')
    partner=ObjectRef('partner','partner','coaster','partner',digest('partner'),ready=True,readiness_source='catalog')
    template=Template('baseline','inter_object_relation_satisfaction-under_small_scale','class','train','house_single_floor','room',
        (target,support,partner),digest('bddl'),{'inst_to_name':{'target':'target_object','support':'support_object','partner':'partner_object'}})
    req={'classes':['class'],'target_scale':[.45]*3,'support_scale':[2,2,1.65]}
    probe={'scene':'house_single_floor','object':'support_object','absolute_scale':[2,2,1.65]}
    profile={'category':'booth','model':'table','asset_sha256':digest('table'),'default_scale':[2,2,1.5]}
    job={'category':'beer_glass','model':'glass','asset_sha256':digest('glass'),'scale':[.45]*3}
    return template,req,probe,profile,job


def test_nonunit_defaults_and_height_only_survive():
    t,*args=fixture();p=m.scale_plan(t,*args)
    assert p.scales['target'].default==(.5,)*3 and p.scales['target'].applied==(.45,)*3
    assert p.scales['support'].default==(2,2,1.5) and p.scales['support'].applied==(2,2,1.65)
    assert p.scales['partner'].default==p.scales['partner'].applied==(1,1,1)
    assert p.perturbed


def test_different_asset_scene_or_coupled_task_rejected():
    t,r,p,s,j=fixture()
    for kind in ['scene','target','support','coupled','horizontal']:
        tt,rr,pp,ss,jj=copy.deepcopy((t,r,p,s,j))
        if kind=='scene':pp['scene']='another_scene'
        if kind=='target':jj['model']='different'
        if kind=='support':ss['asset_sha256']=digest('different')
        if kind=='coupled':tt=replace(tt,objects=(replace(tt.objects[0],category='jigsaw_puzzle_piece'),*tt.objects[1:]));jj['category']='jigsaw_puzzle_piece'
        if kind=='horizontal':rr['support_scale']=pp['absolute_scale']=[2.1,2,1.65]
        with pytest.raises(ContractError):m.scale_plan(tt,rr,pp,ss,jj)


def test_binding_records_applied_state_without_relabeling_it_default():
    t,*args=fixture();plan=m.scale_plan(t,*args)
    t=replace(t,scene_setup={**t.scene_setup,'training_scale_plan_sha256':digest(asdict(plan))})
    job={'training_scale_plan':asdict(plan),'production_released':False}
    with patch.object(m,'bind_one',side_effect=lambda physical,*a:physical) as bound:
        logical,observed=m.bind_training_candidate(t,job,'metadata','code',{}, {}, {})
    physical=bound.call_args.args[0]
    assert physical.objects[0].default_scale==(.45,)*3
    assert logical.objects[0].default_scale==(.5,)*3
    assert observed==plan and observed.perturbed
    assert logical.scene_setup['training_original_defaults_preserved'] is True


def test_binding_rejects_changed_plan_or_actual_scale():
    t,*args=fixture();plan=m.scale_plan(t,*args)
    t=replace(t,scene_setup={**t.scene_setup,'training_scale_plan_sha256':digest(asdict(plan))})
    job={'training_scale_plan':asdict(plan),'production_released':False}
    bad=copy.deepcopy(job);bad['training_scale_plan']['scales']['target']['applied']=[.44]*3
    with pytest.raises(ContractError,match='plan changed'):m.bind_training_candidate(t,bad,'metadata','code',{}, {}, {})
    with patch.object(m,'bind_one',return_value=t),pytest.raises(ContractError,match='applied scale differs'):
        m.bind_training_candidate(t,job,'metadata','code',{}, {}, {})
