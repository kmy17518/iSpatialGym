import json
from pathlib import Path

from scripts.campaign_20261110 import reconcile_consumed_maps as m
from scripts.campaign_20261110 import vast_prebuild as v
from scripts.campaign_20261110.tests.test_vast_map_prebuild import consumed_fixture


def fixture(root):
    manifest, _, _, _, ticket = consumed_fixture(root)
    terminal = Path(ticket['prepared']['path']).parent / 'terminal.json'
    v.write(terminal, {'finished_at': 900, 'started_at': 800})
    document=json.loads(manifest.read_text());document['observed_at']=700;v.write(manifest,document)
    return {'path':str(manifest),'sha256':v.sha(manifest)},ticket,terminal


def test_verified_zero_has_stable_work_times_and_no_downstream(tmp_path):
    ref,ticket,_=fixture(tmp_path);calls=[]
    def reader(*args):
        calls.append(1);return {'status':'valid_zero','native_reader_passed':True,'positive_cells':0,'verified_files':[]}
    out=tmp_path/'callback'
    first=m.reconcile(tmp_path,ref,None,out,reader=reader,timestamp=1000)
    second=m.reconcile(tmp_path,ref,None,out,reader=reader,timestamp=2000)
    assert len(calls)==1 and first['rows'][0]['verification_at']==second['rows'][0]['verification_at']
    assert second['rows'][0]['terminal_at']==900 and second['rows'][0]['manifest_ready_at']==700
    side=json.loads((tmp_path/'infra/coordinator_dispatch/preparation_verification'/(ticket['dedup_key']+'.json')).read_text())
    assert side['status']=='verified_zero' and side['verified_at']==first['rows'][0]['verification_at']
    assert not side['downstream_required']and side['next_check_at']is None


def test_unchanged_failure_is_disposed_without_repeated_reader(tmp_path):
    ref,ticket,_=fixture(tmp_path);calls=[]
    def reader(*args):calls.append(1);raise ValueError('native startup failed')
    for timestamp in [1000,2000]:report=m.reconcile(tmp_path,ref,None,tmp_path/'callback',reader=reader,timestamp=timestamp)
    assert len(calls)==1 and report['summary']=={'invalid_or_failed':1}
    side=json.loads((tmp_path/'infra/coordinator_dispatch/preparation_verification'/(ticket['dedup_key']+'.json')).read_text())
    assert side['status']=='failed' and side['verified_at']is None and 'startup failed'in side['blocker']


def test_crash_after_proof_before_state_resumes_existing_verification(tmp_path):
    ref,ticket,_=fixture(tmp_path);out=tmp_path/'callback'
    def reader(*args):return {'status':'verified_positive','native_reader_passed':True,'positive_cells':3,'verified_files':[]}
    first=m.reconcile(tmp_path,ref,None,out,reader=reader,timestamp=1000)
    (out/'state.json').unlink()
    def never(*args):raise AssertionError('must reuse retained immutable proof')
    second=m.reconcile(tmp_path,ref,None,out,reader=never,timestamp=2000)
    assert second['rows'][0]['verification_at']==first['rows'][0]['verification_at']
    assert second['verified_this_cycle']==0


def test_changed_verified_payload_fails_closed(tmp_path):
    ref,_,_=fixture(tmp_path);payload=tmp_path/'map.npz';payload.write_text('exact grid')
    def reader(*args):return {'status':'verified_positive','verified_files':[{'path':str(payload),'sha256':v.sha(payload)}]}
    out=tmp_path/'callback';m.reconcile(tmp_path,ref,None,out,reader=reader,timestamp=1000)
    payload.write_text('changed grid')
    second=m.reconcile(tmp_path,ref,None,out,reader=reader,timestamp=2000)
    assert second['summary']=={'invalid_or_failed':1}and'Input changed'in second['rows'][0]['blocker']


def test_two_backend_outputs_are_not_silently_merged(tmp_path):
    ref,ticket,_=fixture(tmp_path);job=tmp_path/'vast_job';v.write(job/'result/map_prebuild_result.json',{'finished_at':901})
    route=tmp_path/'route.json';v.write(route,{'source_manifest':ref,'jobs':[{'physical_dedup_key':ticket['dedup_key'],'job_dir':str(job)}]})
    report=m.reconcile(tmp_path,ref,{'path':str(route),'sha256':v.sha(route)},tmp_path/'callback',timestamp=1000)
    assert report['summary']=={'invalid_or_failed':1}and'Two backend'in report['rows'][0]['blocker']
