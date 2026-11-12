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


def test_watch_uses_start_based_cadence_and_stops_without_extra_cycle():
    class Clock:
        now=0
        def monotonic(self):return self.now
        def sleep(self,duration):self.now+=duration
    clock=Clock();starts=[];durations=iter([12,35,12]);stopped=[False]
    def callback():starts.append(clock.now);clock.now+=next(durations)
    m.run_cycles(callback,interval=30,max_cycles=3,stopped=stopped,clock=clock)
    assert starts==[0,30,65]and clock.now==77
    starts=[];clock.now=0
    def stop_callback():starts.append(clock.now);stopped[0]=True
    stopped[0]=False;m.run_cycles(stop_callback,interval=30,max_cycles=10,stopped=stopped,clock=clock)
    assert starts==[0]and clock.now==0


def test_secondary_proof_projection_keeps_private_payload_on_owner(tmp_path, monkeypatch):
    primary=tmp_path/'primary';peer=tmp_path/'secondary';monkeypatch.setattr(m,'account_root',lambda account:peer)
    pp=primary/'prepared.json';sp=peer/'prepared.json';v.write(pp,{'physical_identity':{'height':.7}});v.write(sp,{'physical_identity':{'height':.7}})
    ref=lambda p:{'path':str(p),'sha256':v.sha(p)}
    ticket={'physical_dedup_key':'key','expected_account':'minyeong','prepared':ref(pp)}
    receipt={'expected_account':'shiyuc','prepared':ref(sp),'job_id':'42'}
    assert m.peer_verification(ticket,receipt,'minyeong',{})['status']=='delegated_verification_pending'
    terminal=peer/'terminal.json';v.write(terminal,{'physical_dedup_key':'key','prepared_sha256':v.sha(sp)})
    proof=peer/'reader/verifications/key/proof.json';v.write(proof,{'physical_dedup_key':'key','job_id':'42','native_reader_passed':True,
        'status':'verified_positive','positive_cells':5,'verification_at':100,'verification_started_at':90,'terminal_at':80,
        'terminal':ref(terminal),'verified_files':[{'path':str(peer/'unreadable_private_grid.npz'),'sha256':'owner_verified'}]})
    side=peer/'infra/coordinator_dispatch/preparation_verification/key.json';v.write(side,{'physical_dedup_key':'key','status':'verified_positive','evidence':[ref(proof)]})
    result=m.peer_verification(ticket,receipt,'minyeong',{})
    assert result['status']=='verified_positive'and result['verified_files']==[]
    assert result['verification_at']==100 and len(result['peer_verified_files'])==1
    v.write(sp,{'physical_identity':{'height':.8}});receipt['prepared']=ref(sp)
    import pytest
    with pytest.raises(ValueError,match='physical input identity'):m.peer_verification(ticket,receipt,'minyeong',{})


def test_designated_rest_is_forwarded_to_native_library_reader():
    from types import SimpleNamespace
    calls = []
    schema = SimpleNamespace(load_model=lambda *a, **k: calls.append((a, k)))
    entry = {'category': 'sunflower', 'model': 'exact_model', 'scale': [1., 1., 1.],
             'constraint': 'flower_stem_top', 'rest_hint_wxyz': [.5, .5, .5, .5]}
    m.load_map_library(schema, entry, '/exact/library')
    assert calls[0][1] == {'constraint': 'flower_stem_top', 'rest_hint_wxyz': [.5, .5, .5, .5]}


def test_contact_grid_rejects_wrong_identity_and_corrupt_status():
    from copy import deepcopy
    from types import SimpleNamespace
    import numpy as np
    import pytest
    obj = {'entry': 'exact_model_scale', 'rest_id': 'r0'}
    identity = {'kind': 'push_blade', 'arm': 'right', 'object_key': obj, 'height_above_floor': .8,
                'pose_set': {'digest': 'exact_pose_set'}, 'config': {'edge_distances': [.3, .4]},
                'policy_key': 'policy', 'robot': 'robot'}
    meta = {'schema': 'schema', 'key': 'key', 'kind': 'push_blade', 'arm': 'right', 'plane': 'slab',
            'object': obj, 'library': {'digest': 'lib'}, 'height': {'value': .8}, 'pose_set': identity['pose_set'],
            'config': identity['config'], 'edge_distances': [.3, .4], 'policy_key': 'policy', 'robot': 'robot'}
    grid = np.array([[[1], [0]]], dtype=np.uint8)
    native = SimpleNamespace(meta=deepcopy(meta), cells=np.zeros((1, 2)), yaws=np.zeros(2), classes=['a'], arrays={'s0': grid})
    cm = SimpleNamespace(SCHEMA='schema', map_key=lambda value: 'key', object_key=lambda value: value,
                         KINDS={'push_blade': {'plane': 'slab'}}, STATUS={'feasible': 1, 'no_entry': 0}, STATUS_IK_MARGIN=13)
    cp = SimpleNamespace(DEFAULT=SimpleNamespace(key=lambda arm: 'policy'))
    rm = SimpleNamespace(robot_digest=lambda: 'robot')
    args = (identity, {'object': obj, 'digest': 'lib'}, cm, cp, rm, np)
    assert m.contact_grid_proof(native, *args) == 1
    for field, bad, message in [('pose_set', {'digest': 'other'}, 'height/pose'), ('object', {'entry': 'other'}, 'object identity'),
                                ('config', {'edge_distances': [.2]}, 'configuration'), ('robot', 'other', 'policy/robot')]:
        native.meta = deepcopy(meta); native.meta[field] = bad
        with pytest.raises(ValueError, match=message): m.contact_grid_proof(native, *args)
    native.meta = deepcopy(meta); native.arrays = {'s0': np.array([[[255], [0]]])}
    with pytest.raises(ValueError, match='status grid'): m.contact_grid_proof(native, *args)
    native.arrays = {'s0': np.zeros((1, 1, 1))}
    with pytest.raises(ValueError, match='dimensions'): m.contact_grid_proof(native, *args)


def test_registry_extension_retains_old_entries_and_enforces_bound(tmp_path):
    import pytest
    source = tmp_path / 'registry.json'; pointer = tmp_path / 'pointer.json'
    item = {'manifest': {'path': str(tmp_path / 'tickets.json'), 'sha256': 'fixed'}, 'output': str(tmp_path / 'output')}
    for count in (33, 64, 65):
        v.write(source, {'schema': 'isg-consumed-map-verifier-inputs-v1', 'owner': '/root', 'manifests': [item] * count})
        v.write(pointer, {'path': str(source), 'sha256': v.sha(source)})
        if count <= 64: assert len(m.verifier_registry(tmp_path, pointer)) == count
        else:
            with pytest.raises(ValueError, match='Unbounded'): m.verifier_registry(tmp_path, pointer)
