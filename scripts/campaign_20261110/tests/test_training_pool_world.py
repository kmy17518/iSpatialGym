import copy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scripts.campaign_20261110.contracts import ContractError
from scripts.campaign_20261110.training_pool_world import (
    cold_anchor_transform,scene_contract,isolated_member_scene,verify_pool_isolation)


def test_cold_anchor_rotates_offset_and_returns_to_original_pose():
    first={'position':[1.,2.,0.],'orientation':[0.,0.,0.,1.]}
    anchor={'position':[1.1,2.,.01],'orientation':Rotation.from_euler('x',.01).as_quat().tolist()}
    incoming={'position':[5.,6.,0.],'orientation':Rotation.from_euler('z',np.pi/2).as_quat().tolist()}
    moved=cold_anchor_transform(first,anchor,incoming)
    assert np.allclose(moved['position'],[5.,6.1,.01],atol=1e-12)
    restored=cold_anchor_transform(incoming,moved,first)
    assert np.allclose(restored['position'],anchor['position'],atol=1e-12)
    assert np.allclose(restored['orientation'],anchor['orientation'],atol=1e-12)


def scenes():
    def doc(model,scale):
        name='campaign_pool_'+model
        info=lambda category,model:{'args':{'category':category,'model':model,'fixed_base':False}}
        state={'root_link':{'pos':[0.,0.,1.],'ori':[0.,0.,0.,1.],'lin_vel':[0.,0.,0.],'ang_vel':[0.,0.,0.]}}
        return {'init_info':{'args':{'scene_model':'same_scene'}},
            'objects_info':{'init_info':{'support':info('table','table_a'),name:info('bottle',model)}},
            'state':{'registry':{'object_registry':{'support':copy.deepcopy(state),name:copy.deepcopy(state)}}},
            'metadata':{'task':{'inst_to_name':{'target':name,'support':'support'},'robot_poses':{}},
                'ispatialgym':{'model_pool':{'manifest_sha256':'pool','active_names':[name],
                    'active_properties':{name:{'category':'bottle','model':model,'scale':[scale]*3,'visible':True,
                                               'links':{'base':{'gravity_disabled':False,'collisions':{'shape':True}}}}}}}}}
    return [doc('A',.9),doc('B',1.1),doc('A',.9)]


def test_union_preserves_source_bytes_and_separate_fixed_scale_actors():
    docs=scenes();saved=copy.deepcopy(docs)
    c=scene_contract(docs,['campaign_pool_A','campaign_pool_B','campaign_pool_A'],{'sha256':'pool'})
    assert docs==saved and set(c['initialized_actors'])=={'campaign_pool_A','campaign_pool_B'}
    assert len(c['union_scene']['objects_info']['init_info'])==3
    assert c['active_properties']['campaign_pool_B']['scale']==[1.1]*3
    assert c['production_released'] is False


@pytest.mark.parametrize('change',['support_geometry','scene','unknown_pool','different_manifest','active_hidden'])
def test_undeclared_physical_changes_fail_closed(change):
    docs=scenes();b=docs[1];pool=b['metadata']['ispatialgym']['model_pool']
    if change=='support_geometry':b['objects_info']['init_info']['support']['args']['model']='table_b'
    elif change=='scene':b['init_info']['args']['scene_model']='other'
    elif change=='unknown_pool':
        b['objects_info']['init_info']['campaign_pool_extra']={};b['state']['registry']['object_registry']['campaign_pool_extra']={}
    elif change=='different_manifest':pool['manifest_sha256']='different'
    else:pool['active_properties']['campaign_pool_B']['visible']=False
    with pytest.raises(ContractError):scene_contract(docs,['campaign_pool_A','campaign_pool_B','campaign_pool_A'],{'sha256':'pool'})


def test_each_union_member_preserves_its_exact_scene_and_other_fixed_scale_actor():
    docs=scenes(); original=copy.deepcopy(docs)
    contract=scene_contract(docs,['campaign_pool_A','campaign_pool_B','campaign_pool_A'],{'sha256':'pool'})
    # Distinct accepted source backgrounds must be selected from B itself.
    docs[1]['state']['registry']['object_registry']['support']['root_link']['pos'][0]=.001
    union, isolation=isolated_member_scene(contract,docs[1],'campaign_pool_B')
    assert union['state']['registry']['object_registry']['support']==docs[1]['state']['registry']['object_registry']['support']
    assert union['state']['registry']['object_registry']['campaign_pool_B']==docs[1]['state']['registry']['object_registry']['campaign_pool_B']
    assert union['objects_info']['init_info']['campaign_pool_A']==original[0]['objects_info']['init_info']['campaign_pool_A']
    inactive=isolation['campaign_pool_A'];assert not inactive['active']
    assert inactive['state']['root_link']['pos']==[40.,40.,5.]
    assert inactive['properties']['scale']==[.9]*3 and not inactive['properties']['visible']
    assert inactive['properties']['links']['base']=={'gravity_disabled':True,'collisions':{'shape':False}}
    assert docs[0]==original[0] and docs[2]==original[2]


def isolation_fixture():
    docs=scenes();contract=scene_contract(docs,['campaign_pool_A','campaign_pool_B','campaign_pool_A'],{'sha256':'pool'})
    _,expected=isolated_member_scene(contract,docs[0],'campaign_pool_A')
    observed={name:{'properties':copy.deepcopy(row['properties']),
              'position':row['state']['root_link']['pos'],
              'linear_velocity':[0.,0.,0.],'angular_velocity':[0.,0.,0.],
              'assisted_held':False,'attached_parent':None} for name,row in expected.items()}
    return expected,observed


@pytest.mark.parametrize('fault',['visible','collision','gravity','scale','planner','grasp','binding','motion'])
def test_native_isolation_rejects_each_unreleased_inactive_effect(fault):
    expected,observed=isolation_fixture();other=observed['campaign_pool_B']
    bindings={'target':'campaign_pool_A'};planner=['campaign_pool_A']
    if fault=='visible':other['properties']['visible']=True
    elif fault=='collision':other['properties']['links']['base']['collisions']['shape']=True
    elif fault=='gravity':other['properties']['links']['base']['gravity_disabled']=False
    elif fault=='scale':other['properties']['scale']=[1.,1.,1.]
    elif fault=='planner':planner.append('campaign_pool_B')
    elif fault=='grasp':other['assisted_held']=True
    elif fault=='binding':bindings['old_target']='campaign_pool_B'
    else:other['linear_velocity'][0]=.01
    with pytest.raises(ContractError):verify_pool_isolation(expected,observed,task_bindings=bindings,planner_names=planner)


def test_native_isolation_accepts_only_exact_inactive_properties_and_exclusion():
    expected,observed=isolation_fixture()
    result=verify_pool_isolation(expected,observed,task_bindings={'target':'campaign_pool_A'},planner_names=['campaign_pool_A'])
    assert result['inactive']==['campaign_pool_B'] and result['properties_verified']
