from types import SimpleNamespace
import pytest
from ispatialgym.multienv.campaign_render_cleanup import write_frame,ControlledRenderFailure,ControlledRenderInterrupt


class Writer:
    def __init__(self):
        self.pending=False;self.current_traj_history=[];self.aborted=False;self.abort_calls=0
        self.dataset=SimpleNamespace(has_pending_frames=lambda:self.pending)
    def begin(self,obs):self.current_traj_history=[obs];self.aborted=False
    def step(self,*args):self.pending=True;self.current_traj_history.append(args)
    def abort(self):self.abort_calls+=1;self.pending=False;self.current_traj_history=[];self.aborted=True


def demo():return SimpleNamespace(action=[0,1],reward=[1,1],terminated=[False,True],truncated=[False,False])


@pytest.mark.parametrize('fault,expected',[('failure',ControlledRenderFailure),('interruption',ControlledRenderInterrupt)])
def test_abort_clears_pending_native_writer_then_next_episode_begins(fault,expected):
    w=Writer();d=demo()
    assert write_frame(w,d,0,{'frame':0},fault=fault)==(None,None)
    error,proof=write_frame(w,d,1,{'frame':1},fault=fault)
    assert type(error)is expected and proof['expected_controlled_failure']
    assert proof['trajectory_buffer_empty']and not proof['dataset_pending_frames']and w.abort_calls==1
    assert proof['physics_steps_added']==proof['render_ticks_added']==0
    assert write_frame(w,d,0,{'next':0})==(None,None)and not w.aborted


def test_unexpected_error_still_cleans_up_and_remains_unexpected():
    w=Writer();w.begin({})
    def broken(*args):w.pending=True;raise ValueError('real writer failure')
    w.step=broken
    error,proof=write_frame(w,demo(),1,{},fault='failure')
    assert type(error)is ValueError and not proof['expected_controlled_failure']
    assert not w.pending and w.abort_calls==1


def test_cleanup_failure_cannot_be_reported_successful():
    w=Writer();w.abort=lambda:None
    with pytest.raises(RuntimeError,match='did not clear'):
        write_frame(w,demo(),1,{},fault='interruption')
