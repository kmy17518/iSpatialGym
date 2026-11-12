import json
from pathlib import Path
from dataclasses import asdict
import pytest
from scripts.campaign_20261110.contracts import Template,ObjectRef,ContractError
from scripts.campaign_20261110.tests.test_runtime_suite import put
from scripts.campaign_20261110.runtime_suite_opening_correction import recorded_opening


def fixture(root,task='configuration_matching-articulation_open_small_scale'):
    native=root/'source/template';native.mkdir(parents=True)
    t=Template('template',task,task+'_01','eval_standard','house_single_floor','dining_room_0',
        (ObjectRef('target','target','blender','model','a'*64,joint='hinge',ready=True,readiness_source='source'),),'b'*64)
    joints={'hinge':1.5};runtime={'joint_names':{'target':['hinge']},'planned_goal_joints':{'target':[1.5]}}
    historic={'bucket_strict':True,'fraction_range':[.5,1.],'requested_goal':{'joints':joints},'actual_open_fraction':.75}
    tro={'ispatialgym':{'campaign':{'runtime':runtime}}};summary={'direction':'open','joint':'hinge','campaign_goal_group':historic}
    sources={'tro':put(native/'instances/tro.json',tro),'summary':put(native/'instances/summary.json',summary),
             'sidecar':put(native/'instances/sidecar.json',{}),'job_env':put(root/'source/job.env',{})}
    metadata=put(native/'template_meta.json',{'template_id':'template','template_facts':{'limits':[0.,2.]}})
    plan={'schema':'isg-joint-goal-plan-v1','template_id':'template','target':'target','joint':'hinge',
          'goal_joints':joints,'fraction_range':[0.,.5],'anchor_sources':sources,'budget_seconds':900,'group_id':'old','max_members':3}
    prior={'run_id':'run','class_id':t.class_id,'camera_exempt':False,'frozen_runtime':{'commit':'native'},
           'campaign_config':put(root/'config.json',{'template':asdict(t)}),'catalog_row':put(root/'catalog.json',{}),
           'goal_plan':put(root/'old_plan.json',plan),'effective_limits':{'seconds':900}}
    priorref=put(root/'row.json',prior)
    proposed={**prior,'selected_bucket':1,'goal_plan':put(root/'new_plan.json',{**plan,'group_id':'new','fraction_range':[.5,1.]})}
    e={'class_id':t.class_id,'original_row':priorref,'original_configuration':prior['campaign_config'],
       'original_goal_plan':prior['goal_plan'],'catalog_row':prior['catalog_row'],'frozen_runtime':prior['frozen_runtime'],
       'opening_mismatch':True,'direction':'open','native_checks_changed':False,'source_evidence':sources,
       'native_template_metadata':metadata,'limits':[0.,2.],'goal_fraction':.75,'goal_joint':1.5,
       'historical_group':historic,'authored_range':[0.,.5],'expected_opening_range':[.5,1.]}
    audit={'schema':'isg-current100-articulation-recorded-range-audit-v1','owner':'/root/templates_assets',
           'no_original_mutation':True,'run_id':'run','rows':[e]}
    return put(root/'audit.json',audit),priorref,proposed


def test_recomputes_exact_opening_range_without_touching_sources(tmp_path):
    audit,prior,new=fixture(tmp_path);before={p:p.read_bytes()for p in tmp_path.rglob('*')if p.is_file()}
    proof=recorded_opening(tmp_path,audit,prior,new)
    assert proof['strict_range']==[.5,1.] and proof['fraction']==.75
    assert all(p.read_bytes()==b for p,b in before.items())


@pytest.mark.parametrize('key,value',[('goal_joints',{'hinge':1.8}),('fraction_range',[0.,1.]),
                                     ('budget_seconds',1800),('max_members',4)])
def test_does_not_allow_goal_or_check_changes(tmp_path,key,value):
    a,p,n=fixture(tmp_path);plan=json.loads(Path(n['goal_plan']['path']).read_text());plan[key]=value
    n['goal_plan']=put(Path(n['goal_plan']['path']),plan)
    with pytest.raises(ContractError):recorded_opening(tmp_path,a,p,n)


@pytest.mark.parametrize('key,value',[('goal_fraction',.25),('limits',[0.,4.]),('direction','close'),
                                     ('native_checks_changed',True),('expected_opening_range',[0.,1.])])
def test_audit_claim_cannot_override_native_inputs(tmp_path,key,value):
    a,p,n=fixture(tmp_path);audit=json.loads(Path(a['path']).read_text());audit['rows'][0][key]=value
    with pytest.raises(ContractError):recorded_opening(tmp_path,put(Path(a['path']),audit),p,n)


def test_closing_is_not_covered_by_fixed_opening_proof(tmp_path):
    a,p,n=fixture(tmp_path,'configuration_matching-articulation_close_small_scale')
    with pytest.raises(ContractError,match='opening standard'):recorded_opening(tmp_path,a,p,n)
