from dataclasses import replace
import pytest
from scripts.campaign_20261110 import contracts as c
from scripts.campaign_20261110.joint_goal_eligibility import bind_standard_opening, bind_opening, eligible_buckets
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
    assert next_work(template,{**counts,'evaluation':{1:5}})['groups'] == 5
    assert next_work(template,{**counts,'evaluation':{1:10}}) is None
    assert bind_standard_opening(template,anchor) is template


def test_changed_or_unprobed_goal_fails_closed(tmp_path):
    original,anchor=fixture(tmp_path);template=bind_standard_opening(original,anchor)
    anchor['template_meta']['template_facts']['artsmall_draw']['pairs']=[]
    c.write_json_atomic(tmp_path/'template_meta.json',anchor['template_meta'])
    with pytest.raises(c.ContractError,match='source bytes changed'):eligible_buckets(template)
    with pytest.raises(c.ContractError,match='not physically probed'):bind_standard_opening(original,anchor)


@pytest.mark.parametrize('split,high_quota', [('eval_model',4), ('eval_category',1)])
def test_held_fixed_goal_only_fills_its_actual_strict_bucket(tmp_path,split,high_quota):
    original,anchor=fixture(tmp_path);original=replace(original,split=split)
    template=bind_opening(original,anchor)
    assert eligible_buckets(template)==(1,)
    counts={'evaluation':{},'train_buckets':{},'class_remaining':1000,'task_remaining':2000}
    decision=next_work(template,counts)
    assert decision['phase']==split and decision['bucket']==1
    assert next_work(template,{**counts,'evaluation':{1:high_quota}}) is not None
    total = 9 if split == 'eval_model' else 2
    assert next_work(template,{**counts,'evaluation':{1:total}}) is None
    assert template.objects==original.objects and template.definition_sha256==original.definition_sha256
    with pytest.raises(c.ContractError,match='standard split'):bind_standard_opening(original,anchor)


def test_held_revision_is_explicit_and_does_not_change_legacy_identity(tmp_path):
    from scripts.campaign_20261110.controller_admission import admission_id
    original,anchor=fixture(tmp_path);template=replace(original,split='eval_model')
    (tmp_path/'.snapshot_commit').write_text('a'*40+'\n')
    legacy=admission_id(template,tmp_path,{},'eval_model')
    revised=admission_id(template,tmp_path,{},'eval_model','held_native_opening_v1')
    assert legacy!=revised and legacy==admission_id(template,tmp_path,{},'eval_model')
    for split in ('train','eval_standard'):
        with pytest.raises(c.ContractError,match='unsupported admission author revision'):
            admission_id(template,tmp_path,{},split,'held_native_opening_v1')
    with pytest.raises(c.ContractError,match='evaluated small articulation'):
        bind_opening(replace(template,split='train'),anchor)
