from types import SimpleNamespace as NS

import numpy as np
import pytest

from ispatialgym.sampling import campaign_runtime as runtime
from ispatialgym.sampling import multiply_isg as multiply
from ispatialgym.sampling.v2.tasks.disembedding import Disembedding


@pytest.mark.parametrize('side',['left','right'])
@pytest.mark.parametrize('split',['eval_standard','eval_model','eval_category','train'])
def test_campaign_driver_keeps_exact_feasible_side_before_native_setup(monkeypatch,side,split):
    monkeypatch.setenv('ISPATIALGYM_CAMPAIGN_CONFIG','/owned/config.json')
    monkeypatch.setattr(runtime,'configuration',lambda:({'split':split},NS(task='disembedding'),None,None))
    gp=NS(subtask='disembedding',variant='disembedding-private',generation={key:side for key in multiply.EVAL_ONLY_GENERATION_KEYS})
    gp.generation['distractor_gap']=.01
    multiply.prepare_train_generation(gp)
    assert gp.generation=={'destination_side':side,'distractor_gap':.01}
    sampler=NS(g=lambda key:gp.generation.get(key),gp=gp,template_facts={})
    for seed in range(16):
        assert Disembedding._destination_side(sampler,np.random.default_rng(seed))==side
    assert sampler.template_facts['destination_side']=={'side':side,'source':'generation.destination_side'}


def test_legacy_training_still_draws_both_sides(monkeypatch):
    monkeypatch.delenv('ISPATIALGYM_CAMPAIGN_CONFIG',raising=False)
    gp=NS(subtask='disembedding',variant='disembedding-old',generation={'destination_side':'right'})
    multiply.prepare_train_generation(gp)
    sampler=NS(g=lambda key:gp.generation.get(key),gp=gp,template_facts={})
    assert {Disembedding._destination_side(sampler,np.random.default_rng(i)) for i in range(16)}=={'left','right'}


def test_campaign_other_tasks_keep_train_draw_policy(monkeypatch):
    monkeypatch.setenv('ISPATIALGYM_CAMPAIGN_CONFIG','/owned/config.json')
    monkeypatch.setattr(runtime,'configuration',lambda:({},NS(task='object_scaling'),None,None))
    gp=NS(subtask='object_scaling',generation={key:1 for key in multiply.EVAL_ONLY_GENERATION_KEYS})
    multiply.prepare_train_generation(gp)
    assert gp.generation=={}


def test_campaign_without_explicit_side_still_draws_both(monkeypatch):
    monkeypatch.setenv('ISPATIALGYM_CAMPAIGN_CONFIG','/owned/config.json')
    monkeypatch.setattr(runtime,'configuration',lambda:({},NS(task='disembedding'),None,None))
    gp=NS(subtask='disembedding',variant='disembedding-unpinned',generation={})
    multiply.prepare_train_generation(gp)
    sampler=NS(g=lambda key:gp.generation.get(key),gp=gp,template_facts={})
    assert {Disembedding._destination_side(sampler,np.random.default_rng(i)) for i in range(16)}=={'left','right'}


def test_invalid_campaign_configuration_fails_before_generation_mutation(monkeypatch,tmp_path):
    path=tmp_path/'config.json';path.write_text('{}')
    monkeypatch.setenv('ISPATIALGYM_CAMPAIGN_CONFIG',str(path))
    gp=NS(subtask='disembedding',generation={'destination_side':'right'})
    with pytest.raises(KeyError):multiply.prepare_train_generation(gp)
    assert gp.generation=={'destination_side':'right'}


def test_main_train_plan_uses_campaign_generation_preparation(monkeypatch,tmp_path):
    monkeypatch.setenv('ISPATIALGYM_CAMPAIGN_CONFIG','/owned/config.json')
    monkeypatch.setattr(runtime,'configuration',lambda:({},NS(task='disembedding'),None,None))
    gp=NS(subtask='disembedding',variant='disembedding-private',generation={'destination_side':'right'},floor_only=lambda:False)
    choice=NS(scene='scene',rooms=['room'],template_id='native')
    monkeypatch.setattr(multiply,'load_template',lambda path:{'variant':gp.variant})
    monkeypatch.setattr(multiply,'problem_for_template',lambda *args:gp)
    monkeypatch.setattr(multiply,'choice_for_template',lambda *args:choice)
    args=NS(out=str(tmp_path/'train'),template=[str(tmp_path/'eval/variant/native')],path_buckets=None,
            customs=None,bddl_root=None,seed=1,split='train',keep_template_scene=True,start_index=1,
            instances=1,only_scene=None,only_room=None,plan=True)
    assert multiply.main_train(args)==0
    assert gp.generation['destination_side']=='right'
