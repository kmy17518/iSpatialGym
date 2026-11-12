import csv,json
from pathlib import Path
import pytest
from scripts.campaign_20261110 import runtime_suite_allocations as A,runtime_suite_replays as R,runtime_validation_metrics as M


def put(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v));return M.reference(p)


def test_allocation_requires_exact_ledger_runtime_and_reserved_range(tmp_path):
    source=tmp_path/'out/job';source.mkdir(parents=True);native=tmp_path/'code/frozen';native.mkdir(parents=True)
    ledger=tmp_path/'manifests/allocation.tsv';ledger.parent.mkdir();a={'jobid':'123','job_dir':str(source),'code':str(native),'start':'100','n':'3','key':'class','ledger':str(ledger)}
    with ledger.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(a),delimiter='\t');w.writeheader();w.writerow(a)
    row={'dispatch_token':'t','fresh_start':100,'fresh_count':3,'class_id':'class','frozen_runtime':{'path':str(native)}}
    receipt={'status':'submitted','token':'t','allocation':a}
    assert A.allocation(row,receipt,tmp_path)==a
    for k,v in [('start','101'),('code',str(tmp_path/'other')),('key','other')]:
        with pytest.raises(ValueError):A.allocation(row,{**receipt,'allocation':{**a,k:v}},tmp_path)


def test_member_gate_must_bind_exact_source_runtime_tro_and_demo(tmp_path):
    source=tmp_path/'out/job';demo=source/'demo/task__100/demo.hdf5';demo.parent.mkdir(parents=True);demo.write_bytes(b'demo')
    tro=source/'inst/task__100-tro_state.json';tro.parent.mkdir();tro.write_bytes(b'tro')
    runtime=tmp_path/'code/frozen';row={'frozen_runtime':{'path':str(runtime)}}
    g={'source_job':str(source),'snapshot':str(runtime),'demo':str(demo),'instance':str(tro),'demo_sha256':M.reference(demo)['sha256'],'tro_sha256':M.reference(tro)['sha256']}
    p=tmp_path/'gate.json';put(p,g);assert A.gate_member(p,row,source)[0]=='task__100'
    demo.write_bytes(b'changed')
    with pytest.raises(ValueError,match='inputs changed'):A.gate_member(p,row,source)


def test_exact_member_successes_preserve_canonical_first_and_bounds(tmp_path,monkeypatch):
    from scripts.campaign_20261110 import acceptance
    monkeypatch.setattr(acceptance,'report_passed',lambda *a:None)
    source=tmp_path/'out/job';native=tmp_path/'code/frozen';row={'fresh_start':100,'fresh_count':3,'frozen_runtime':{'path':str(native)}}
    for idx in (100,101,102):
        iid='task__'+str(idx);put(source/'demo'/iid/'meta.json',{'success':True,'instance_id':iid,'provenance':{'isg_root':str(native)}})
        (source/'demo'/iid/'demo.hdf5').write_bytes(str(idx).encode());put(source/'inst'/(iid+'-tro_state.json'),{})
    first,members=R.successful_members(source,row)
    assert first=='task__100' and [m['instance_id'] for m in members]==['task__100','task__101','task__102']
    row['fresh_count']=2
    with pytest.raises(ValueError,match='fresh range'):R.successful_members(source,row)


def test_regression_outcome_never_counts_as_accepted_eval(tmp_path):
    row={'run_id':'r','class_id':'c','frozen_runtime':{},'campaign_config':{},'group_id':'g','fresh_start':1,'fresh_count':3,'camera_exempt':False}
    members=[{'instance_id':f'i{i}','demo_passed':True,'physics_replay_passed':True,'physics_replay_applicability':'own_member_replay'} for i in (1,2)]
    quota={'eligible':False,'quota_ineligible':True,'consumed':0,'accepted':{}}
    result={**row,'schema':'isg-current-runtime-class-acceptance-v1','row':{},'allocation':{},'members':members,'quota':quota,'physical_validation_passed':True,'validation_passed':True,'status':'regression_validation_passed',
        'strict_review':put(tmp_path/'review.json',{'members':members}),'regression_audit':put(tmp_path/'audit.json',{'validation_passed':True}),'quota_evidence':put(tmp_path/'quota.json',quota)}
    value=M.strict_outcomes(result,row,{},{});assert value['accepted_eval_instances']==0 and value['replay_passed_instances']==2 and value['completed_goal_groups']==1
    assert M.rates({'observed_allocated_gpu_hours':2,'cost_complete':True},value)['gpu_hours_per_accepted_eval_instances'] is None
    result['quota']['consumed']=1;result['quota_evidence']=put(tmp_path/'quota2.json',result['quota'])
    with pytest.raises(ValueError,match='regression consumed'):M.strict_outcomes(result,row,{},{} )


def test_canonical_suite_handoff_retains_existing_attempt_owners():
    from scripts.campaign_20261110.controller_validation import isolated_suite_replay_owner
    decision={'allowed':True,'category':'runtime_suite_validation','runtime_suite':{'row':{'path':'exact','sha256':'bound'}}}
    assert isolated_suite_replay_owner(decision,{})
    assert not isolated_suite_replay_owner(decision,{'replay_job_id':'123'})
    assert not isolated_suite_replay_owner(decision,{'vast_migration':{'status':'prepared'}})
    assert not isolated_suite_replay_owner({**decision,'allowed':False},{})
    assert not isolated_suite_replay_owner({**decision,'category':'ordinary_validation'},{})
