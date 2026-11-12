import json
from pathlib import Path
import pytest
from scripts.campaign_20261110 import training_sequential_status as status


def fixture(tmp_path):
    batch=tmp_path/'batch';d=batch/'sequential_verification_r1/physics/fresh_A';d.mkdir(parents=True)
    p=d/'request.json';p.write_text('{}');r=status.ref(p)
    (d/'preflight.json').write_text('{}')
    (d/'submission.json').write_text(json.dumps({'request':r,'jobs':[{'request':r,'job_id':'123','returncode':0,'control':'fresh_A'}]}))
    obs={'observed_at':'actual_clock','source':{'path':'raw','sha256':'exact'}}
    return batch,d,obs


def test_submission_does_not_become_execution_or_pass(tmp_path):
    b,d,o=fixture(tmp_path)
    row=status.control(tmp_path,b,'physics','fresh_A',{},o)
    assert row['status']=='awaiting_scheduler_observation'and row['scheduler_state']is None
    assert row['verification_status']=='not_verified'
    actual={'123':{'state':'COMPLETED','exit_code':'0:0'}}
    row=status.control(tmp_path,b,'physics','fresh_A',actual,o)
    assert row['status']=='awaiting_strict_verification'and row['verification_status']=='not_verified'
    assert row['scheduler_observed_at']=='actual_clock'


def test_current_exact_probe_binding_required_for_verified_state(tmp_path):
    b,d,o=fixture(tmp_path);(d/'output').mkdir();p=d/'output/probe.json';p.write_text('{"entries":[]}')
    v={'status':'verified','stages':{'physics':{'fresh_A':{'probe':status.ref(p)}}}}
    (d.parent.parent/'verification.json').write_text(json.dumps(v))
    row=status.control(tmp_path,b,'physics','fresh_A',{'123':{'state':'COMPLETED','exit_code':'0:0'}},o)
    assert row['verification_status']=='verified'
    p.write_text('{"entries":[{}]}')
    row=status.control(tmp_path,b,'physics','fresh_A',{'123':{'state':'COMPLETED','exit_code':'0:0'}},o)
    assert row['status']=='evidence_error'and row['verification_status']=='not_verified'


def test_all_eleven_rows_and_failed_execution_remain_explicit(tmp_path):
    b,d,o=fixture(tmp_path)
    value=status.project(tmp_path,b,{'123':{'state':'FAILED','exit_code':'1:0'}},o)
    assert len(value['controls'])==11 and value['verified_controls']==0
    first=value['controls'][0];assert first['status']=='execution_failed'
    assert value['controls'][-1]['status']=='awaiting_request'
    v=json.loads((d/'submission.json').read_text());v['request']['sha256']='wrong';(d/'submission.json').write_text(json.dumps(v))
    assert status.control(tmp_path,b,'physics','fresh_A',{},o)['status']=='evidence_error'


def test_rendering_only_proof_cannot_verify_physics(tmp_path):
    b,d,o=fixture(tmp_path);rd=b/'sequential_verification_r1/rendering/fresh_A';rd.mkdir(parents=True)
    for name in ('request.json','preflight.json'):(rd/name).write_bytes((d/name).read_bytes())
    request=status.ref(rd/'request.json');(rd/'submission.json').write_text(json.dumps({'request':request,'jobs':[{'request':request,'job_id':'124','returncode':0,'control':'fresh_A'}]}))
    (rd/'output').mkdir();p=rd/'output/probe.json';p.write_text('{"entries":[{}]}')
    value={'schema':'isg-sequential-rendering-verification-v1','status':'verified','verified_stages':['rendering'],
           'full_matrix_verified':False,'stages':{'rendering':{'fresh_A':{'probe':status.ref(p)}}}}
    (b/'sequential_verification_r1/rendering_verification.json').write_text(json.dumps(value))
    scheduler={'123':{'state':'COMPLETED','exit_code':'0:0'},'124':{'state':'COMPLETED','exit_code':'0:0'}}
    assert status.control(tmp_path,b,'rendering','fresh_A',scheduler,o)['verification_status']=='verified'
    assert status.control(tmp_path,b,'physics','fresh_A',scheduler,o)['verification_status']=='not_verified'


def test_prospective_attempt_preserves_original_status_and_separate_jobs(tmp_path):
    b,d,o=fixture(tmp_path);directory=b/'sequential_anchor_r2';rows=[]
    for name in status.PHYSICS:
        p=directory/'physics'/name/'request.json';p.parent.mkdir(parents=True);p.write_text('{}')
        rows.append({'stage':'physics','control':name,'request':status.ref(p)})
    (directory/'requests.json').write_text(json.dumps({'records':rows}))
    value=status.project(tmp_path,b,{'123':{'state':'COMPLETED','exit_code':'0:0'}},o)
    assert len(value['controls'])==11 and len(value['prospective_controls'])==6
    assert value['controls'][0]['job_id']=='123'and value['prospective_controls'][0]['job_id']is None
    assert value['prospective_verified_controls']==0 and value['verified_controls']==0


def test_known_failure_is_scoped_to_exact_probe(tmp_path):
    b,d,o=fixture(tmp_path);(d/'output').mkdir();p=d/'output/probe.json';p.write_text('{"entries":[]}')
    failure={'schema':'isg-sequential-control-failures-v1','controls':{'physics/fresh_A':{
        'probe':status.ref(p),'status':'blocked_by_failed_matrix','reason':'other control failed',
        'evidence':[status.ref(d/'request.json')],'next_action':'diagnose'}}}
    path=d.parent.parent/'verification_failure_observations.json';path.write_text(json.dumps(failure))
    row=status.control(tmp_path,b,'physics','fresh_A',{'123':{'state':'COMPLETED','exit_code':'0:0'}},o)
    assert row['status']=='blocked_by_failed_matrix'and row['verification_status']=='blocked'
    p.write_text('{"entries":[{}]}')
    assert status.control(tmp_path,b,'physics','fresh_A',{},o)['status']=='evidence_error'
