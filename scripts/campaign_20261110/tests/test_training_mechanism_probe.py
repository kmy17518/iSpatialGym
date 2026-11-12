import pytest
from scripts.campaign_20261110.contracts import ContractError
from scripts.campaign_20261110.training_mechanism_probe import scope, ORDERS, DELTA


def request(control):
    return {'schema':'isg-retained-mechanism-replay-request-v1','campaign_id':'campaign_20261110',
        'control':control,'physical_delta':dict(DELTA),'benchmark_credit':0,'production_released':False,
        'max_native_episodes':len(ORDERS[control]),'timeout_seconds':1800}


@pytest.mark.parametrize('control',list(ORDERS))
def test_exact_retained_source_controls_are_finite_and_never_new_draws(control):
    assert scope(request(control)) == ORDERS[control] and set(scope(request(control))) == {0}


@pytest.mark.parametrize('field,value',[('production_released',True),('benchmark_credit',1),
    ('timeout_seconds',36000),('max_native_episodes',3),('control','ABA'),
    ('physical_delta',{**DELTA,'additional_physics_steps':1})])
def test_control_cannot_expand_episodes_release_scope_or_change_physics(field,value):
    with pytest.raises(ContractError):scope({**request('AA'),field:value})
