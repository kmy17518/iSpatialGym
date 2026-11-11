import copy

import pytest

from scripts.campaign_20261110.training_attachment_dependencies import (
    coupled_profile, permitted, shared_support_requests,
)


def obj(category='lid', model='male', scale=(.65,.65,.65)):
    return dict(category=category,model=model,default_scale=list(scale),asset_sha256='a'*64,
                ready=True,readiness_source='native inventory')


def test_pair_scales_share_relative_draw_and_preserve_original_defaults():
    target=obj();partner=obj('stockpot','female',(.7,.8,.9))
    original=copy.deepcopy((target,partner));p=coupled_profile(target,partner)
    assert p==coupled_profile(target,partner)
    assert (target,partner)==original
    for slot,factor in enumerate(p['sampled_uniform_factors']):
        assert .8<=factor<=1.2
        for role,ref in [('target',target),('partner',partner)]:
            assert p['absolute_scales'][role][slot]==[round(v*factor,4) for v in ref['default_scale']]
    assert p['maximum_absolute_scales']['target']==[.78]*3
    assert p['mating_geometry_status'].startswith('requires fresh native')


def test_pair_identity_changes_when_either_port_changes():
    a,b=obj(),obj('stockpot','female');p=coupled_profile(a,b)
    b['model']='another';assert coupled_profile(a,b)['id']!=p['id']


@pytest.mark.parametrize('field,value',[('ready',False),('readiness_source',''),('asset_sha256','')])
def test_unready_provenance_rejected(field,value):
    a=obj();a[field]=value
    with pytest.raises(ValueError):permitted(a,dict(category_holdout_categories=[],train_exclude_models=[]))


@pytest.mark.parametrize('policy',[dict(category_holdout_categories=['lid'],train_exclude_models=[]),
                                  dict(category_holdout_categories=[],train_exclude_models=['lid/male'])])
def test_holdout_assets_rejected(policy):
    with pytest.raises(ValueError):permitted(obj(),policy)


def test_ten_support_candidates_do_not_claim_layout_or_edge_validation():
    profiles={'profiles':{'p':obj('table','support')},'bindings':{
        str(i):dict(profile_id='p',scene='room',name='table'+str(i),shared_anchor=True)for i in range(10)}}
    policy=dict(category_holdout_categories=[],train_exclude_models=[])
    rows=shared_support_requests(profiles,['attachment01'],['pair'],policy)
    assert len(rows)==10 and all(r['production_admitted'] is False for r in rows)
    assert all(r['maximum_layout_fit'].startswith('pending_native') for r in rows)
    assert all(r['usable_edges'].startswith('pending_native') for r in rows)
    profiles['bindings'].pop('0')
    with pytest.raises(ValueError):shared_support_requests(profiles,['attachment01'],['pair'],policy)
