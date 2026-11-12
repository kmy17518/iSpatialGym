import numpy as np
import pandas as pd
import pytest
from scripts.campaign_20261110.contracts import ContractError
from scripts.campaign_20261110.training_pool_export import exact_episode_rows


def fixture():
    data = pd.DataFrame({'action':[np.array([1.,2.],dtype='float32'),np.array([3.,4.],dtype='float32')],
        'next.reward':np.array([0.,1.],dtype='float32'),'next.terminated':[False,True],
        'next.truncated':[False,False],'observation.state':[np.array([5.]),np.array([6.])],
        'timestamp':np.array([0.,1/30],dtype='float32'),'frame_index':[0,1],
        'episode_index':[0,0],'index':[0,1],'task_index':[0,0]})
    source={'action':np.stack(data.action),'reward':data['next.reward'].to_numpy(),
            'terminated':data['next.terminated'].to_numpy(),'truncated':data['next.truncated'].to_numpy()}
    return data,source


def test_exact_export_retains_cold_arrays_and_source_actions_with_new_global_indexes():
    cold,source=fixture();actual=cold.copy();actual['episode_index']=1;actual['index']=[2,3]
    proof=exact_episode_rows(actual,cold,source,1,2)
    assert proof['frames']==2 and proof['source_action_reward_done_exact']


@pytest.mark.parametrize('mutation',['action','proprio','index','extra_prefix','source_action'])
def test_changed_arrays_or_aborted_rows_cannot_pass_export(mutation):
    cold,source=fixture();actual=cold.copy(deep=True)
    if mutation=='action':actual.at[0,'action']=np.array([10.,2.],dtype='float32')
    elif mutation=='proprio':actual['observation.state']=[np.array([5.]),np.array([6.000000001])]
    elif mutation=='index':actual['index']=[1,2]
    elif mutation=='extra_prefix':actual=pd.concat([actual.iloc[:1],actual],ignore_index=True)
    elif mutation=='source_action':source['action']=source['action']+1
    with pytest.raises(ContractError):exact_episode_rows(actual,cold,source,0,0)
