import copy
import json
from dataclasses import asdict, replace
from pathlib import Path
import pytest
from scripts.campaign_20261110 import runtime_suite_correction as c, runtime_suite_retry as r
from scripts.campaign_20261110.contracts import ContractError, Template, digest
from scripts.campaign_20261110.tests.test_runtime_suite import put
from scripts.campaign_20261110.tests.test_runtime_suite_author import disembedding_anchor
from scripts.campaign_20261110.tests.test_runtime_suite_retry import proposed


def correction_fixture(root):
    native=root/'source/template';native.mkdir(parents=True)
    partial,anchor=disembedding_anchor(native)
    objects=tuple(replace(o,ready=True,readiness_source='exact source') for o in partial.objects)
    poolref=put(root/'pool.json',{'exact':'full pool'})
    pool={'manifest':poolref['path'],'manifest_sha256':poolref['sha256']}
    template=Template('template','disembedding','disembedding_01','eval_standard','house_single_floor',
        'dining_room_0',objects,'a'*64,{'binding_pool':pool})
    anchor['template_meta']['template_id']='template'
    metadata=put(native/'template_meta.json',anchor['template_meta'])
    sources={name:put(native/'instances'/(name+'.json'),anchor[name]) for name in ('tro','summary')}
    sources.update(sidecar=put(native/'instances/sidecar.json',{}),job_env=put(root/'source/env.json',{}))
    old={'schema':'isg-tray-goal-plan-v1','group_id':'original','template_id':'template',
         'crowdedness_range':[0,.3],'anchor_sources':sources,'goal_pose':[1,2,3],'budget_seconds':1800}
    prior={'run_id':'run','class_id':'disembedding_01','camera_exempt':False,
           'frozen_runtime':{'commit':'fd95'},'selected_bucket':0,
           'campaign_config':put(root/'config.json',{'template':asdict(template)}),
           'catalog_row':put(root/'catalog.json',{'class_id':'disembedding_01'}),
           'goal_plan':put(root/'old_plan.json',old)}
    priorref=put(root/'row.json',prior)
    new={**old,'group_id':'corrected','crowdedness_range':[.3,.45]}
    proposed={**prior,'selected_bucket':1,'goal_plan':put(root/'new_plan.json',new)}
    profile=anchor['template_meta']['template_facts']['campaign_coverage_profile']
    active=[asdict(o) for o in objects]
    evidence={'class_id':'disembedding_01','original_row':priorref,'original_configuration':prior['campaign_config'],
      'original_goal_plan':prior['goal_plan'],'catalog_row':prior['catalog_row'],'frozen_runtime':prior['frozen_runtime'],
      'status':'input_authoring_mismatch','pool_geometry_goal_change':False,'bucket_threshold_change':False,
      'native_sampler_change':False,'source_evidence':sources,'native_template_metadata':metadata,
      'native_coverage_profile':profile,'native_coverage_profile_sha256':digest(profile),'active_objects':active,
      'active_objects_sha256':digest(active),'measured_coverage':.32,'measured_bucket':1,'binding_pool':pool,
      'binding_pool_manifest':poolref,'expected_strict_range':[.3,.45],'historical_strict_range':[.3,.45],
      'historical_actual_coverage':.32,'authored_strict_range':[0,.3]}
    audit={'schema':'isg-current100-disembedding-recorded-range-audit-v1','owner':'/root/templates_assets',
           'native_sampler_changed':False,'original_rows_unchanged':True,'run_id':'run','rows':[evidence]}
    return put(root/'audit.json',audit),priorref,proposed


def test_exact_range_recomputed_from_original_source_without_mutation(tmp_path):
    audit,prior,new=correction_fixture(tmp_path)
    before={p:p.read_bytes() for p in tmp_path.rglob('*.json')}
    proof=c.recorded_range(tmp_path,audit,prior,new)
    assert proof['measurement']['strict_range']==[.3,.45]
    assert all(p.read_bytes()==b for p,b in before.items())


@pytest.mark.parametrize('field,value',[('crowdedness_range',[0,1]),('goal_pose',[9,9,9]),
                                       ('budget_seconds',2700),('max_members',4)])
def test_correction_is_not_range_or_goal_relaxation(tmp_path,field,value):
    audit,prior,new=correction_fixture(tmp_path)
    plan=json.loads(Path(new['goal_plan']['path']).read_text());plan[field]=value
    new['goal_plan']=put(Path(new['goal_plan']['path']),plan)
    with pytest.raises(ContractError):c.recorded_range(tmp_path,audit,prior,new)


@pytest.mark.parametrize('field,value',[('measured_coverage',.2),('expected_strict_range',[0,1]),
    ('measured_bucket',0),('historical_actual_coverage',.2),('native_sampler_change',True),
    ('original_row',{}),('binding_pool_manifest',{})])
def test_audit_label_cannot_override_native_source(tmp_path,field,value):
    audit,prior,new=correction_fixture(tmp_path)
    body=json.loads(Path(audit['path']).read_text());body['rows'][0][field]=value
    audit=put(Path(audit['path']),body)
    with pytest.raises(ContractError):c.recorded_range(tmp_path,audit,prior,new)


def test_source_mutation_and_bucket_only_relabel_rejected(tmp_path):
    audit,prior,new=correction_fixture(tmp_path)
    new['selected_bucket']=0
    with pytest.raises(ContractError):c.recorded_range(tmp_path,audit,prior,new)
    new['selected_bucket']=1
    (tmp_path/'source/template/instances/tro.json').write_text('{}')
    with pytest.raises(ContractError,match='bytes changed'):c.recorded_range(tmp_path,audit,prior,new)


def test_only_float64_sum_roundoff_is_allowed_in_audit_scalar(tmp_path):
    import math
    audit,prior,new=correction_fixture(tmp_path)
    body=json.loads(Path(audit['path']).read_text())
    body['rows'][0]['measured_coverage']=math.nextafter(.32,1.)
    assert c.recorded_range(tmp_path,put(Path(audit['path']),body),prior,new)
    body['rows'][0]['measured_coverage']=.32+1e-12
    with pytest.raises(ContractError):c.recorded_range(tmp_path,put(Path(audit['path']),body),prior,new)


def test_correction_terminal_path_accepts_no_worker_only_with_complete_terminal_proof(tmp_path):
    base,prior,allocation,rowref,proposal=proposed(tmp_path)
    # This private phase is reached only after recorded_range checked all inputs.
    (tmp_path/'out/job/job.env').unlink();(tmp_path/'out/job/sample.log').unlink()
    result=r.failure(tmp_path,prior,allocation,[],correction_ref={'path':'verified audit','sha256':'x'})
    assert result['failure_kind']=='input_authoring_mismatch' and not result['physical_failure_inferred']
    body=json.loads(Path(allocation['path']).read_text())
    terminal=json.loads(Path(body['terminal_evidence']['path']).read_text())
    raw=json.loads((base/'raw.json').read_text());raw['squeue_stdout']='123|RUNNING'
    terminal['jobs'][0]['raw']=put(base/'raw.json',raw)
    body['terminal_evidence']=put(base/'terminal.json',terminal)
    with pytest.raises(ContractError,match='active'):
        r.failure(tmp_path,prior,put(Path(allocation['path']),body),[],correction_ref={'path':'verified','sha256':'x'})


def test_ordinary_retry_can_change_descriptive_bucket_with_unchanged_goal(tmp_path):
    base,prior,allocation,rowref,proposal=proposed(tmp_path)
    body=json.loads(Path(proposal['path']).read_text());row=json.loads(Path(rowref['path']).read_text())
    row['selected_bucket']=1;body['row']=put(Path(rowref['path']),row)
    assert r.prepare(tmp_path,put(Path(proposal['path']),body))
