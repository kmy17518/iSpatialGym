from pathlib import Path
import json
import pytest
from scripts.campaign_20261110 import training_sequential_stream as stream
from scripts.campaign_20261110.contracts import ContractError,write_json_atomic


def test_native_failure_does_not_authorize_collect_or_matrix(tmp_path,monkeypatch):
    batch=tmp_path/'batch';batch.mkdir();author=tmp_path/'author.py';author.write_text('raise RuntimeError("must not run")')
    request=batch/'requests.json';write_json_atomic(request,{'campaign_id':'campaign_20261110','records':[{'candidate_id':'a'},{'candidate_id':'b'}]})
    config={'batch':str(batch),'requests':stream.ref(request),'prepared':str(tmp_path/'prepared'),'authorizer':stream.ref(author),'authorizations':str(tmp_path/'auth'),'runtime':str(tmp_path/'runtime'),'campaign_started_at':0}
    def failed(*args):raise ContractError('native canary did not write exactly one successful sample')
    monkeypatch.setattr(stream,'native_evidence',failed)
    monkeypatch.setattr(stream,'prepare_matrix',lambda *a:pytest.fail('failed source cannot author matrix'))
    state=stream.run_cycle(tmp_path,config,{})
    assert {r['status'] for r in state['records'].values()}=={'engineering_blocked'}
    assert not(tmp_path/'auth').exists()


def test_matrix_does_not_replace_pinned_sources(tmp_path):
    batch=tmp_path/'batch';directory=batch/'sequential_verification_r1';directory.mkdir(parents=True)
    write_json_atomic(directory/'requests.json',{'runtime':'native','canaries':[{'path':'old','sha256':'a'}],'records':[]})
    with pytest.raises(ContractError,match='source identity changed'):
        stream.prepare_matrix(tmp_path,batch,[{'path':'new','sha256':'b'}],Path('native'),0)


def test_snapshot_preserves_previous_exact_bytes(tmp_path):
    p=tmp_path/'state.json';stream.publish_snapshot(p,{'phase':'native'})
    first=p.read_bytes();key=stream.file_digest(p)
    stream.publish_snapshot(p,{'phase':'replay'})
    assert (tmp_path/'state_history'/(key+'.json')).read_bytes()==first
    assert json.loads(p.read_text())['phase']=='replay'


def test_changed_pair_preserves_failed_prior_source_without_refreshing_its_age(tmp_path):
    path=tmp_path/'reviews.json'
    stream.publish_reviews(path,{'observed_at':'first','records':{'A':{'status':'failed_own_replay','failure_evidence':{'gate':'original'}}}})
    stream.publish_reviews(path,{'observed_at':'second','records':{'A2':{'status':'awaiting_native'}}})
    value=stream.read(path)
    assert value['records']['A']=={'status':'failed_own_replay','failure_evidence':{'gate':'original'},'observed_at':'first','current_pair_member':False}
    assert value['records']['A2']['observed_at']=='second' and value['records']['A2']['current_pair_member'] is True


def test_replay_failure_retains_task_success_and_measured_qa_failure(tmp_path):
    report=tmp_path/'meta.json';metric={'measured':True,'triggered':True,'report':{'max_occluded':.6301,'threshold':.5}}
    stream.immutable(report,{'qa':{'metrics':{'goal_occlusion':metric}}})
    gate={'success':False,'validated_at':'2026-11-12T07:19:45Z','replay_rc':0,
          'native_quality':{'reasons':['native_hard_qa:goal_occlusion'],'native_report':str(report),'native_report_sha256':stream.file_digest(report)},
          'result':{'qa_hard':['goal_occlusion'],'geometric_success':True,'symbolic_success':True,'n_action_steps':738}}
    path=tmp_path/'gate.json';stream.immutable(path,gate)
    evidence=stream.replay_failure(tmp_path,path,gate)
    assert evidence['geometric_success'] is True and evidence['hard_qa']==['goal_occlusion']
    assert evidence['goal_occlusion']==metric and evidence['gate']==stream.ref(path)
    report.write_text('{}')
    with pytest.raises(ContractError,match='changed'):stream.replay_failure(tmp_path,path,gate)


def test_failed_own_replay_cannot_publish_or_author_matrix(tmp_path,monkeypatch):
    batch=tmp_path/'batch';batch.mkdir();author=tmp_path/'author.py';author.write_text('never run')
    prepared=tmp_path/'prepared';prepared.mkdir();auth=tmp_path/'auth';runtime=tmp_path/'runtime'
    rows=[];allocations=[]
    for i,cid in enumerate(('A','B'),1):
        job=tmp_path/cid;job.mkdir();(job/'demo/member').mkdir(parents=True)
        stream.immutable(job/'demo/member/meta.json',{'success':True})
        row={'candidate_id':cid,'start':i,'count':1,'class_id':'disembedding_07','config_path':str(tmp_path/(cid+'.json')),'config_sha256':cid,'native_runtime':str(runtime),'output':str(job/'native')};rows.append(row)
        allocations.append({'candidate':cid,'job_dir':str(job),'original_index':i,'job_id':str(i)})
    request=batch/'requests.json';stream.immutable(request,{'campaign_id':'campaign_20261110','records':rows})
    for row,allocation in zip(rows,allocations):
        cid=row['candidate_id'];p=prepared/(cid+'.json')
        stream.immutable(p,{'candidate_id':cid,'start':row['start'],'count':1,'configuration':{'path':row['config_path'],'sha256':cid},'runtime':str(runtime),'source_request':stream.ref(request),'job_dir':allocation['job_dir']})
        stream.immutable(auth/cid/'authorization_receipt.json',{'prepared':stream.ref(p),'job_dir':allocation['job_dir']})
        stream.immutable(tmp_path/'validation/open_loop/disembedding_07'/allocation['job_id']/'gate.json',{'success':False,'native_quality':{'reasons':['goal_occlusion']},'result':{'qa_hard':['goal_occlusion']}})
    stream.immutable(prepared/'launched.json',{'jobs':allocations})
    monkeypatch.setattr(stream,'native_evidence',lambda *args:{'native':'passed'})
    monkeypatch.setattr(stream,'publish_canary',lambda *a,**kw:pytest.fail('failed replay cannot publish'))
    monkeypatch.setattr(stream,'prepare_matrix',lambda *a,**kw:pytest.fail('failed replay cannot author matrix'))
    cfg={'batch':str(batch),'requests':stream.ref(request),'prepared':str(prepared),'authorizer':stream.ref(author),'authorizations':str(auth),'runtime':str(runtime),'campaign_started_at':0}
    state=stream.run_cycle(tmp_path,cfg,{})
    assert all(r['status']=='failed_own_replay' and r['failure_evidence']['hard_qa']==['goal_occlusion'] for r in state['records'].values())


def test_prospective_pair_retains_A_and_exact_new_B_without_relabeling_failure(tmp_path):
    old=tmp_path/'old.json';new=tmp_path/'new.json'
    stream.immutable(old,{'campaign_id':'campaign_20261110','records':[{'candidate_id':'A','start':1},{'candidate_id':'failed_B','start':2}]})
    stream.immutable(new,{'campaign_id':'campaign_20261110','records':[{'candidate_id':'B_retry','start':3}]})
    original=old.read_bytes();cfg={'members':[{'requests':stream.ref(old),'candidate_id':'A'},{'requests':stream.ref(new),'candidate_id':'B_retry'}]}
    document,refs=stream.selected_requests(tmp_path,cfg)
    assert [r['start']for r in document['records']]==[1,3]and old.read_bytes()==original
    new.write_text('{}')
    with pytest.raises(ContractError,match='changed'):stream.selected_requests(tmp_path,cfg)


def test_matrix_binds_normal_and_cleanup_rendering_to_one_renderer(tmp_path,monkeypatch):
    native=tmp_path/'code/native';renderer=tmp_path/'code/renderer';checker=tmp_path/'code/checker'
    for directory,label in ((native,'native'),(renderer,'render'),(checker,'checker')):
        (directory/'scripts').mkdir(parents=True);(directory/'ispatialgym').mkdir();(directory/'.snapshot_commit').write_text(label)
    job=tmp_path/'job';job.mkdir();(job/'job.env').write_text('source bytes')
    reviews=[stream.immutable(tmp_path/(name+'.json'),{'identity':name})for name in ('A','B')]
    sources=[{'instance_id':name,'source_job':str(job),'export_record':{'validation_key':{'code_sha256':stream.file_digest(native/'.snapshot_commit')}}}for name in ('A','B')]
    monkeypatch.setattr(stream,'validated_canary',lambda root,reference,started:sources[reviews.index(reference)])
    monkeypatch.setattr(stream,'inputs',lambda root,p:(stream.read(p),[s['export_record']for s in sources]))
    audited=[]
    def audit(root,request,records,runtime):
        assert request['probe_runtime_sha256']==stream.file_digest(runtime/'.snapshot_commit')
        audited.append((request.get('sequential_render_only'),runtime))
        return request['runtime_compatibility']
    monkeypatch.setattr(stream,'verify_runtime_compatibility',audit)
    batch=tmp_path/'batch';manifest=stream.read(stream.prepare_matrix(tmp_path,batch,reviews,native,0,renderer,checker)['path'])
    assert len(manifest['records'])==9
    assert {r['runtime']for r in manifest['records']if r['stage']=='physics'}=={str(checker)}
    assert {r['runtime']for r in manifest['records']if r['stage']=='rendering'}=={str(renderer)}
    assert sum(flag is True for flag,runtime in audited)==3
    assert all(stream.read(r['request']['path'])['sequential_checker_only'] is True for r in manifest['records']if r['stage']=='physics')
    cleanup=stream.read(stream.prepare_render_cleanup(tmp_path,batch,reviews,native,renderer,stream.ref(job/'job.env'))['path'])
    assert len(cleanup['records'])==2 and {r['runtime']for r in cleanup['records']}=={str(renderer)}
    proof=stream.read(manifest['records'][-1]['runtime_proof']['path'])
    assert proof['permitted_execution_kinds']==['kinematic_observation_rendering']
    with pytest.raises(ContractError,match='source identity changed'):
        stream.prepare_matrix(tmp_path,batch,reviews,native,0,native)
