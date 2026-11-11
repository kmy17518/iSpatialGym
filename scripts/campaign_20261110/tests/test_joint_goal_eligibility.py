from dataclasses import replace
import pytest
from scripts.campaign_20261110 import contracts as c
from scripts.campaign_20261110.joint_goal_eligibility import bind_standard_opening, eligible_buckets
from scripts.campaign_20261110.controller_groups import next_work
from scripts.campaign_20261110.tests.test_goal_groups import GoalGroupControllerTests


def fixture(tmp_path):
    _,_,_,original,_=GoalGroupControllerTests().fixture(tmp_path)
    target=replace(original.objects[0],role='target',joint='lid')
    template=replace(original,task='configuration_matching-articulation_open_small_scale',objects=(target,))
    tro={'ispatialgym':{'object_bindings':{target.instance:{'category':target.category,'model':target.model}},
        'campaign':{'runtime':{'joint_names':{target.instance:['lid']},'planned_goal_joints':{target.instance:[.575]}}}}}
    metadata={'template_facts':{'direction':'open','joint':'lid','limits':[0.,1.],
        'artsmall_draw':{'pairs':[{'q_init':0.,'q_goal':.5},{'q_init':0.,'q_goal':.575}]}}}
    c.write_json_atomic(tmp_path/'tro.json',tro);c.write_json_atomic(tmp_path/'template_meta.json',metadata)
    return template,{'template_dir':str(tmp_path),'template_meta':metadata,'sources':{'tro':{'path':str(tmp_path/'tro.json'),'sha256':c.file_digest(tmp_path/'tro.json')}}}


def test_fixed_high_goal_never_targets_low_bucket_or_spills_training(tmp_path):
    original,anchor=fixture(tmp_path);template=bind_standard_opening(original,anchor)
    assert eligible_buckets(template)==(1,)
    assert template.objects==original.objects and template.fingerprint!=original.fingerprint
    counts={'evaluation':{},'train_buckets':{},'class_remaining':1000,'task_remaining':2000}
    assert next_work(original,counts)['bucket']==0
    assert next_work(template,counts)['bucket']==1
    assert next_work(template,{**counts,'evaluation':{1:5}}) is None
    assert bind_standard_opening(template,anchor) is template


def test_changed_or_unprobed_goal_fails_closed(tmp_path):
    original,anchor=fixture(tmp_path);template=bind_standard_opening(original,anchor)
    anchor['template_meta']['template_facts']['artsmall_draw']['pairs']=[]
    c.write_json_atomic(tmp_path/'template_meta.json',anchor['template_meta'])
    with pytest.raises(c.ContractError,match='source bytes changed'):eligible_buckets(template)
    with pytest.raises(c.ContractError,match='not physically probed'):bind_standard_opening(original,anchor)
