from types import SimpleNamespace
import pytest
from ispatialgym.sampling import campaign_tray_goals as C
from ispatialgym.sampling.campaign_held_goals import GoalPlanError


def hint():
    return {'sidecar_sha256': 'a'*64, 'tro_sha256': 'b'*64, 'report_sha256': 'c'*64,
            'initial_base_pose2d': [1, 2, .3], 'initial_lateral_exact_first': True,
            'manipulation': {'base_pose2d': [1,2,.3], 'focus': [2,3,.8], 'translate': .25, 'tilt': -.53}}


def test_exact_first_then_ordinary_proposal():
    h=C.reference_start_proposal(SimpleNamespace(task='disembedding', class_id='disembedding_10', split='eval_standard', scene_setup={'reference_start_proposal':hint()}))
    class R:
        calls=0
        def uniform(self,a,b):self.calls+=1;return (a+b)/2+.01
    r=R()
    assert C.initial_lateral_proposal(h,(1,2,.3),(-.27,.24),-.167,0,r)==-.167
    assert r.calls==0
    assert C.initial_lateral_proposal(h,(1,2,.3),(-.27,.24),-.167,1,r)==pytest.approx(-.157)
    assert r.calls==1


@pytest.mark.parametrize('class_id,split',[('disembedding_01','eval_standard'),('disembedding_10','train'),('disembedding_10','eval_model')])
def test_exact_scope(class_id,split):
    with pytest.raises(GoalPlanError,match='scope'):
        C.reference_start_proposal(SimpleNamespace(task='disembedding',class_id=class_id,split=split,scene_setup={'reference_start_proposal':hint()}))


def test_no_clamping_or_unbound_anchor():
    h={'initial_base_pose2d':(1,2,.3),'initial_lateral_exact_first':True}
    with pytest.raises(GoalPlanError,match='native bounds'):C.initial_lateral_proposal(h,(1,2,.3),(0,1),-.1,0,None)
    with pytest.raises(GoalPlanError,match='audited'):C.initial_lateral_proposal(h,(1,3,.3),(0,1),.5,0,None)


def test_default_stays_random():
    r=SimpleNamespace(uniform=lambda a,b:.123)
    assert C.initial_lateral_proposal(None,(1,2,.3),(-1,1),0,0,r)==.123


def test_missing_provenance_rejected():
    h=hint();h.pop('report_sha256')
    with pytest.raises(GoalPlanError,match='report_sha256'):
        C.reference_start_proposal(SimpleNamespace(task='disembedding',class_id='disembedding_10',split='eval_standard',scene_setup={'reference_start_proposal':h}))
