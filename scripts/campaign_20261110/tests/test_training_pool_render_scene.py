import copy
from types import SimpleNamespace as N
import pytest
from scripts.campaign_20261110.contracts import ContractError
from scripts.campaign_20261110.training_pool_render_scene import scene_union, member_scene, observe_visibility


def fixtures():
    documents=[];properties={name:{'category':'bottle','model':name,'scale':[s,s,s]}for name,s in [('A',.9),('B',1.1)]}
    for index,name in enumerate(('A','B')):
        root={'pos':[index,0,1],'ori':[0,0,0,1],'lin_vel':[0,0,0],'ang_vel':[0,0,0]}
        initial={'robot':{'args':{'position':[index,0,0],'orientation':[0,0,0,1],'model':'sameRobot'}},
                 'support':{'args':{'model':'sameSupport'}},name:{'args':properties[name]}}
        documents.append({'objects_info':{'init_info':initial},'state':{'registry':{'object_registry':{n:{'root_link':copy.deepcopy(root)}for n in initial}}},
            'init_info':{'args':{'scene_file':name+'.json','scene_model':'sameHouse','load_room_instances':['sameRoom']}},
            'metadata':{'ispatialgym':{'model_pool':{'active_names':[name],'active_properties':{name:properties[name]}}}}})
    return documents,{'initialized_actors':{'A':{},'B':{}},'active_properties':copy.deepcopy(properties)}


def test_union_keeps_source_geometry_and_poses_private_and_uses_one_canonical_inventory():
    docs,physical=fixtures();before=copy.deepcopy(docs);contract=scene_union(docs,['A','B'],'robot',physical)
    a,park_a=member_scene(contract,docs[0],'A');b,park_b=member_scene(contract,docs[1],'B')
    assert a['objects_info']['init_info']==b['objects_info']['init_info']
    assert list(a['objects_info']['init_info'])==['robot','support','A','B']
    assert park_a=={'B':[41.,40.,5.]} and park_b=={'A':[40.,40.,5.]}
    assert b['state']['registry']['object_registry']['B']==docs[1]['state']['registry']['object_registry']['B']
    assert docs==before


@pytest.mark.parametrize('changed',['support','robot','room','scale'])
def test_unmeasured_geometry_boundaries_are_rejected(changed):
    docs,physical=fixtures()
    if changed=='support':docs[1]['objects_info']['init_info']['support']['args']['model']='differentSupport'
    elif changed=='robot':docs[1]['objects_info']['init_info']['robot']['args']['model']='differentRobot'
    elif changed=='room':docs[1]['init_info']['args']['load_room_instances']=['otherRoom']
    else:docs[1]['metadata']['ispatialgym']['model_pool']['active_properties']['B']['scale']=[2,2,2]
    with pytest.raises(ContractError):scene_union(docs,['A','B'],'robot',physical)


def test_native_visibility_and_parking_are_both_required():
    docs,physical=fixtures();contract=scene_union(docs,['A','B'],'robot',physical);_,parking=member_scene(contract,docs[0],'A')
    objects={name:N(name=name,visible=name=='A',category='bottle',model=name,scale=physical['active_properties'][name]['scale'],
                    get_position_orientation=lambda n=name:([0,0,0]if n=='A'else[41,40,5],[0,0,0,1]))for name in ('A','B')}
    scene=N(object_registry=lambda key,name:objects[name]);assert not observe_visibility(scene,contract,'A',parking)['B']['visible']
    objects['B'].visible=True
    with pytest.raises(ContractError,match='visibility'):observe_visibility(scene,contract,'A',parking)


def test_native_robot_init_without_spawn_fields_is_valid_but_partial_pose_is_not():
    docs,physical=fixtures()
    for doc in docs:
        args=doc['objects_info']['init_info']['robot']['args'];args.pop('position');args.pop('orientation')
    assert scene_union(docs,['A','B'],'robot',physical)['robot_name']=='robot'
    docs[1]['objects_info']['init_info']['robot']['args']['position']=[0,0,0]
    with pytest.raises(ContractError,match='partial'):scene_union(docs,['A','B'],'robot',physical)
