import pytest
from scripts.campaign_20261110.training_batch_probe import episode_order


def test_default_remains_distinct_a_b_a():
    assert episode_order({},'physics_action_open_loop')==[0,1,0]
    assert episode_order({},'kinematic_observation_rendering')==[0,1,0]


def test_repeated_a_requires_explicit_nonrelease_diagnostic():
    request={'diagnostic_control':'A_A','diagnostic_only':True,'production_released':False}
    assert episode_order(request,'physics_action_open_loop')==[0,0]
    for changed in ({'diagnostic_only':False},{'production_released':True},{'diagnostic_control':'A_B'}):
        with pytest.raises(ValueError):episode_order({**request,**changed},'physics_action_open_loop')
    with pytest.raises(ValueError):episode_order(request,'kinematic_observation_rendering')
