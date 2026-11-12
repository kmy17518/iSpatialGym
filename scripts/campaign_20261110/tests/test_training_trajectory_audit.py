import copy
import numpy as np
import pytest
from scripts.campaign_20261110.training_trajectory_audit import compare_arrays
from scripts.campaign_20261110.contracts import ContractError


def traces():
    pose=np.tile([0.,0.,0.,0.,0.,0.,1.],(5,1))
    return {'action':np.zeros((5,23)),'joint_positions':np.zeros((5,28)),
            'joint_velocities':np.zeros((5,28)),
            **{k:pose.copy() for k in ['base_pose','eef_pose_left','eef_pose_right','object_pose/target']}}


def test_exact_motion_and_antipodal_quaternions_match():
    a=traces();b=copy.deepcopy(a);b['object_pose/target'][:,3:]*=-1
    result=compare_arrays(a,b)
    assert result['status']=='matched' and result['frames_compared']==5
    assert result['production_released'] is False


@pytest.mark.parametrize('key,column,delta',[('joint_positions',2,2e-5),
    ('joint_velocities',3,2e-4),('object_pose/target',0,.003)])
def test_late_drift_is_not_hidden_by_matching_first_action(key,column,delta):
    a=traces();b=copy.deepcopy(a);b[key][4,column]=delta
    result=compare_arrays(a,b);assert result['status']=='mismatch'
    metric=result['metrics'][key if key.startswith('joint_') else key+'/position_m']
    assert metric['first_frame_exceeding']==4 and metric['frames_exceeding']==1


def test_orientation_uses_existing_angular_bound():
    a=traces();b=copy.deepcopy(a)
    b['object_pose/target'][3,3:]=[0.,0.,np.sin(1e-4),np.cos(1e-4)]
    assert compare_arrays(a,b)['metrics']['object_pose/target/orientation_rad']['passed'] is False


def test_changed_actions_and_short_ordinary_episode_are_rejected():
    a=traces();b=copy.deepcopy(a);b['action'][2,1]=1e-8
    with pytest.raises(ContractError,match='actions differ'):compare_arrays(a,b)
    short={k:v[:1] for k,v in a.items()}
    with pytest.raises(ContractError,match='frame count'):compare_arrays(a,short)
    result=compare_arrays(a,short,prefix=True)
    assert result['controlled_abort_prefix'] and result['frames_compared']==1


def test_missing_fields_and_nonfinite_measurements_fail_closed():
    a=traces();b=copy.deepcopy(a);b.pop('joint_velocities')
    with pytest.raises(ContractError,match='inventory'):compare_arrays(a,b)
    b=copy.deepcopy(a);b['joint_velocities'][4,0]=np.nan
    with pytest.raises(ContractError,match='finite'):compare_arrays(a,b)
