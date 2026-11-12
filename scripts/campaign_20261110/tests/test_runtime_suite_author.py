import json
from types import SimpleNamespace
import pytest
from scripts.campaign_20261110.runtime_suite_author import preserved_environment,group_range,immutable,checked


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
        assert group_range(SimpleNamespace(task=task),counts)==(edges[2],2)


def test_immutable_exact_hash(tmp_path):
    path=tmp_path/"proof.json";r=immutable(path,{"id":1});assert checked(r)=={"id":1}
    assert immutable(path,{"id":1})==r
    with pytest.raises(AssertionError):immutable(path,{"id":2})
    path.write_text(json.dumps({"id":2}))
    with pytest.raises(AssertionError):checked(r)
