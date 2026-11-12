from copy import deepcopy
import pytest
from scripts.campaign_20261110.audit_floating_arm_results import validate_arm
from scripts.campaign_20261110.contracts import ContractError

def fixture():
    p={'row':{'validation_override':{'arm':'left'},'job':{'held_tasks':['mental_rotation']}}}
    lib={'arm':'left','policy':{'arm':'left'},'policy_key':'k','held_tasks':['mental_rotation'],'grasps':[{'name':'g'}]}
    val={'config':{'arm':'left'},'policy_key':'k','validator':'isg-floating-validator/2','tasks':['mental_rotation'],
         'trials':[{'candidate':'g','task':'mental_rotation','verdict':'valid','motions_executed':1,
                    'motions':[{'executed':True,'held':True,'outcome':'ok'}]}]}
    return p,lib,val,{'arm':'left'}

def test_valid_left_motion_is_admitted():
    assert validate_arm(*fixture())=='left'

@pytest.mark.parametrize('bad',['arm','policy','task','trial','motion'])
def test_right_arm_or_unsupported_motion_cannot_satisfy_left_dependency(bad):
    p,l,v,c=fixture()
    if bad=='arm':v['config']['arm']='right'
    elif bad=='policy':l['policy']['arm']='right'
    elif bad=='task':v['tasks']=['object_scaling']
    elif bad=='trial':v['trials'][0]['verdict']='failed'
    else:v['trials'][0]['motions'][0]['held']=False
    with pytest.raises(ContractError):validate_arm(p,l,v,c)
