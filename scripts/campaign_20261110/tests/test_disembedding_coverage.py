from dataclasses import replace
import pytest
from scripts.campaign_20261110 import contracts as c
from scripts.campaign_20261110.disembedding_coverage import bind_standard_coverage, eligible_buckets
from scripts.campaign_20261110.controller_groups import next_work
from scripts.campaign_20261110.tests.test_standard_disembedding_pool import source


def fixture(tmp_path):
    template = source()
    row = lambda obj: {'instance':obj.instance,'category':obj.category,'model':obj.model,
                       'scale':list(obj.default_scale),'area_m2':.01}
    objects = [o for o in template.objects if o.role in ('target','distractor')]
    profile = {'schema':'isg-native-tray-coverage-profile-v1','target':row(next(o for o in objects if o.role=='target')),
               'distractors':[row(o) for o in objects if o.role=='distractor'],'source_floor_area_m2':.2}
    metadata = {'template_facts':{'campaign_coverage_profile':profile}}
    c.write_json_atomic(tmp_path/'template_meta.json',metadata)
    return template, {'template_meta':metadata,'template_dir':str(tmp_path)}


def test_measured_low_coverage_never_schedules_impossible_middle_or_spills_training(tmp_path):
    original, anchor = fixture(tmp_path); bound = bind_standard_coverage(original,anchor)
    assert original.objects == bound.objects and original.fingerprint != bound.fingerprint
    assert eligible_buckets(bound) == (0,)
    counts = {'evaluation':{},'train_buckets':{},'class_remaining':1000,'task_remaining':10000}
    assert next_work(bound,counts)['bucket'] == 0
    assert next_work(bound,{**counts,'evaluation':{0:3}}) is None
    assert next_work(original,counts)['bucket'] == 1
    assert bind_standard_coverage(bound,anchor) is bound


def test_changed_native_evidence_models_and_boundaries_fail_closed(tmp_path):
    template, anchor = fixture(tmp_path); bound = bind_standard_coverage(template,anchor)
    profile = anchor['template_meta']['template_facts']['campaign_coverage_profile']
    profile['target']['model'] = 'another'
    c.write_json_atomic(tmp_path/'template_meta.json',anchor['template_meta'])
    with pytest.raises(c.ContractError,match='source bytes changed'):eligible_buckets(bound)
    with pytest.raises(c.ContractError,match='model differs'):bind_standard_coverage(template,anchor)
    profile['target']['model'] = next(o.model for o in template.objects if o.role=='target')
    profile['source_floor_area_m2'] = .1
    c.write_json_atomic(tmp_path/'template_meta.json',anchor['template_meta'])
    with pytest.raises(c.ContractError,match='strict bucket boundary'):bind_standard_coverage(template,anchor)
