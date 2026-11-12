import json
from types import SimpleNamespace
import pytest
from scripts.campaign_20261110.runtime_suite_author import preserved_environment,group_range,immutable,checked,recorded_opening_range


def test_preserve_behavior_replace_execution():
    old={"ISGE_CODE":"old","START":"12","N":"8","ISPATIALGYM_CAMPAIGN_GOAL_QUEUE":"oldqueue",
         "ISGF_SCRIPTS":"oldscript","ISPATIALGYM_PUSH_HEEL_TIGHT":"1","ISG_CUSTOMS":"exactcustoms",
         "ISPATIALGYM_CAMPAIGN_OBSTACLE_GOAL":"1","ISGF_GOAL_OCCLUSION":"on","KEEP_SCENE":"exact scene",
         "ISPATIALGYM_REACH_MAPS":"exactmap","ISGF_GRASPS":"v2"}
    kept,replaced=preserved_environment(old)
    assert set(kept)=={"ISPATIALGYM_PUSH_HEEL_TIGHT","ISG_CUSTOMS","ISPATIALGYM_CAMPAIGN_OBSTACLE_GOAL",
                       "ISGF_GOAL_OCCLUSION","KEEP_SCENE","ISPATIALGYM_REACH_MAPS"}
    assert set(kept)|set(replaced)==set(old)


def test_buckets_keep_approved_thresholds():
    from scripts.campaign_20261110.contracts import task_quotas
    for task,edges in [("object_scaling",[(.05,.10),(.10,.15),(.15,.30)]),("mental_rotation",[(30,45),(45,75),(75,120)])]:
        q=next(q for q in task_quotas()if q.task==task)
        counts={str(i):n for i,n in enumerate(q.standard_buckets)};counts["2"]-=1
        assert group_range(SimpleNamespace(task=task),counts)==((edges[0][0],edges[-1][1]),0)


def test_immutable_exact_hash(tmp_path):
    path=tmp_path/"proof.json";r=immutable(path,{"id":1});assert checked(r)=={"id":1}
    assert immutable(path,{"id":1})==r
    with pytest.raises(AssertionError):immutable(path,{"id":2})
    path.write_text(json.dumps({"id":2}))
    with pytest.raises(AssertionError):checked(r)


def test_recorded_opening_goal_keeps_its_strict_bucket():
    template = SimpleNamespace(task="configuration_matching-articulation_open_large_scale", objects=[SimpleNamespace(role="target",instance="fixture",joint="hinge")])
    anchor = {"template_meta":{"template_facts":{"limits":[0.,2.]}},
              "joint_names":{"fixture":["hinge"]},"planned_goal_joints":{"fixture":[1.8]}}
    bounds, bucket = recorded_opening_range(template, anchor)
    assert bounds[0] <= .9 <= bounds[1]
    assert anchor["planned_goal_joints"]["fixture"] == [1.8]
    anchor["planned_goal_joints"]["fixture"] = [3.]
    with pytest.raises(AssertionError):recorded_opening_range(template, anchor)


def disembedding_anchor(tmp_path, coverage=.32):
    from copy import deepcopy
    from scripts.campaign_20261110.contracts import ObjectRef
    from scripts.campaign_20261110.runtime_suite_author import ref
    objects = tuple(ObjectRef(instance=n, role=r, category='piece', model=m, asset_sha256='a'*64)
                    for n, r, m in [('target', 'target', 'aaa'), ('distractor', 'distractor', 'bbb')])
    template = SimpleNamespace(task='disembedding', split='eval_standard', class_id='disembedding_01', objects=objects)
    profile = {'schema':'isg-native-tray-coverage-profile-v1','source_floor_area_m2':1.,
               'target':{'instance':'target','category':'piece','model':'aaa','scale':[1.,1.,1.],'area_m2':.1},
               'distractors':[{'instance':'distractor','category':'piece','model':'bbb','scale':[1.,1.,1.],'area_m2':coverage-.1}]}
    bounds = [0,.3] if coverage < .3 else [.3,.45] if coverage < .45 else [.45,1]
    group = {'group_id':'source_group','plan_sha256':'b'*64,'bucket_strict':True,
             'crowdedness_range':bounds,'actual_crowdedness':coverage}
    anchor = {'template_dir':str(tmp_path),'template_meta':{'template_facts':{'campaign_coverage_profile':profile}},
              'summary':{'campaign_goal_group':deepcopy(group)},
              'tro':{'ispatialgym':{'campaign':{'class_id':template.class_id,'split':'eval_standard',
                                              'runtime':{'goal_group':deepcopy(group)}}}},
              'sources':{}, 'planned_goal_poses':{'target':[[1,2,3],[0,0,0,1]]}}
    (tmp_path/'template_meta.json').write_text(json.dumps(anchor['template_meta']))
    for name in ('tro','summary'):
        path=tmp_path/(name+'.json');path.write_text(json.dumps(anchor[name]));anchor['sources'][name]=ref(path)
    return template,anchor


def rewrite_anchor_bytes(anchor):
    from pathlib import Path
    from scripts.campaign_20261110.runtime_suite_author import ref
    root=Path(anchor['template_dir'])
    (root/'template_meta.json').write_text(json.dumps(anchor['template_meta']))
    for name in ('tro','summary'):
        path=root/(name+'.json');path.write_text(json.dumps(anchor[name]));anchor['sources'][name]=ref(path)


@pytest.mark.parametrize('coverage,expected',[(.22,((0,.3),0)),(.32,((.3,.45),1)),(.55,((.45,1),2))])
def test_recorded_disembedding_preserves_exact_selection_and_goals(tmp_path,coverage,expected):
    from copy import deepcopy
    from scripts.campaign_20261110.runtime_suite_author import recorded_disembedding_range,recorded_disembedding_evidence
    template,anchor=disembedding_anchor(tmp_path,coverage);before=deepcopy(anchor)
    assert recorded_disembedding_range(template,anchor)==expected
    evidence=recorded_disembedding_evidence(template,anchor)
    assert evidence['coverage']==pytest.approx(coverage)
    assert evidence['pool_geometry_goal_changed'] is False
    assert anchor==before


@pytest.mark.parametrize('change', ['model','scale','missing_instance','area','floor','boundary','historical_range',
                                    'historical_actual','not_strict','runtime_disagrees','class','group_identity'])
def test_recorded_disembedding_rejects_incompatible_proof(tmp_path,change):
    from scripts.campaign_20261110.runtime_suite_author import recorded_disembedding_range
    template,anchor=disembedding_anchor(tmp_path)
    profile=anchor['template_meta']['template_facts']['campaign_coverage_profile']
    group=anchor['summary']['campaign_goal_group']
    if change=='model':profile['target']['model']='other'
    elif change=='scale':profile['target']['scale']=[.9,.9,.9]
    elif change=='missing_instance':profile['distractors']=[]
    elif change=='area':profile['target']['area_m2']=float('nan')
    elif change=='floor':profile['source_floor_area_m2']=0
    elif change=='boundary':profile['distractors'][0]['area_m2']=.2
    elif change=='historical_range':group['crowdedness_range']=[0,.3]
    elif change=='historical_actual':group['actual_crowdedness']=.2
    elif change=='not_strict':group['bucket_strict']=False
    elif change=='runtime_disagrees':anchor['tro']['ispatialgym']['campaign']['runtime']['goal_group']['crowdedness_range']=[0,.3]
    elif change=='class':anchor['tro']['ispatialgym']['campaign']['class_id']='disembedding_02'
    elif change=='group_identity':group['plan_sha256']='c'*64
    rewrite_anchor_bytes(anchor)
    if change in ('boundary','historical_range','not_strict','runtime_disagrees'):
        assert recorded_disembedding_range(template,anchor)[1] == 1
    else:
        with pytest.raises((AssertionError,ValueError)):
            recorded_disembedding_range(template,anchor)


def test_recorded_disembedding_static_profile_and_sampled_projection_are_distinct(tmp_path):
    from scripts.campaign_20261110.runtime_suite_author import recorded_disembedding_evidence
    template,anchor=disembedding_anchor(tmp_path,.32)
    for group in (anchor['summary']['campaign_goal_group'],
                  anchor['tro']['ispatialgym']['campaign']['runtime']['goal_group']):
        group['actual_crowdedness']=.299
    rewrite_anchor_bytes(anchor)
    evidence=recorded_disembedding_evidence(template,anchor)
    assert evidence['coverage']==pytest.approx(.32)
    assert evidence['historical_actual_crowdedness']==.299


@pytest.mark.parametrize('source',['tro','summary','metadata'])
def test_recorded_disembedding_checks_saved_bytes(tmp_path,source):
    from scripts.campaign_20261110.runtime_suite_author import recorded_disembedding_range
    template,anchor=disembedding_anchor(tmp_path)
    path=tmp_path/('template_meta.json' if source=='metadata' else source+'.json')
    path.write_text('{}')
    with pytest.raises((AssertionError,ValueError)):
        recorded_disembedding_range(template,anchor)


@pytest.mark.parametrize('joint,names,values,upper,expected',[
    ('j_link0',['j_link0'],[0.501782238483429],0.872664749622345,((.5,1.),1)),
    ('j_right_lid',['j_left_lid','j_right_lid'],[0.,0.7382742166519165],2.356194257736206,((0.,.5),0)),
])
def test_recorded_small_opening_actual_source_goals(joint,names,values,upper,expected):
    # Exact cwkvib/cdmmwy anchors from the immutable current100 range audit.
    # The second source has two joints: use the designated lid, not joint zero.
    from copy import deepcopy
    template=SimpleNamespace(task='configuration_matching-articulation_open_small_scale',
                             objects=[SimpleNamespace(role='target',instance='target',joint=joint)])
    anchor={'template_meta':{'template_facts':{'limits':[0.,upper]}},
            'joint_names':{'target':names},'planned_goal_joints':{'target':values}}
    before=deepcopy(anchor)
    assert recorded_opening_range(template,anchor)==expected
    assert anchor==before


def test_recorded_opening_range_does_not_choose_closing_start_bucket():
    template=SimpleNamespace(task='configuration_matching-articulation_close_large_scale')
    with pytest.raises(AssertionError,match='requires opening'):
        recorded_opening_range(template,{})
