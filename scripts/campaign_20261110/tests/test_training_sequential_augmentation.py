from copy import deepcopy
import pytest
from scripts.campaign_20261110.contracts import ContractError
from scripts.campaign_20261110.training_sequential_augmentation import one_world,compare_physics_reference
from scripts.campaign_20261110.contracts import digest


def observed(sequence):
    state={'objects':{},'lights':{'lamp':1200.},'skybox':{'intensity':1000.,'color':[1.,1.,1.]}}
    context={'physics_steps_added':0,'dynamic_roots':{'target':{'mass':2.,'sleep_threshold':.001,'solver_position_iteration_count':16,'solver_velocity_iteration_count':1}}}
    return {'process_session':'native:1','scene_runtime_id':4,'sim_runtime_id':5,'sequence':sequence,
            'loaded_state':state,'loaded_state_sha256':digest(state),'native_dynamics_context':context,
            'native_dynamics_context_sha256':digest(context)}


def test_constructor_count_and_native_ids_both_required():
    proof={'world_construction':{'method':'omnigibson.Environment','attempted':1,'completed':1}}
    rows=[observed(1),observed(2),observed(3)]
    assert one_world(proof,rows)=='native:1'
    rows[2]['scene_runtime_id']=6
    with pytest.raises(ContractError,match='rebuilt'):one_world(proof,rows)
    rows[2]['scene_runtime_id']=4;proof['world_construction']['completed']=2
    with pytest.raises(ContractError,match='constructor'):one_world(proof,rows)
    proof['world_construction']['completed']=1;rows[2]['sequence']=2
    with pytest.raises(ContractError,match='order'):one_world(proof,rows)


def test_independent_reference_checks_final_lighting_and_reset_mass():
    a={'boundaries':[observed(i)for i in range(3)],'step':{'after':observed(4)}}
    b=deepcopy(a);compare_physics_reference(a,b)
    b['boundaries'][-1]['loaded_state']['lights']['lamp']=1300
    with pytest.raises(ContractError,match='intensity'):compare_physics_reference(a,b)
    b=deepcopy(a);b['boundaries'][0]['native_dynamics_context']['dynamic_roots']['target']['mass']=3.
    b['boundaries'][0]['native_dynamics_context_sha256']=digest(b['boundaries'][0]['native_dynamics_context'])
    with pytest.raises(ContractError,match='mass'):compare_physics_reference(a,b)
