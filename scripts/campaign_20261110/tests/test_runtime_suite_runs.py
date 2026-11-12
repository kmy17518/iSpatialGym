import copy
import json
from pathlib import Path
import pytest
from scripts.campaign_20261110 import runtime_suite as S, production_gate as G, runtime_suite_paths as P
from scripts.campaign_20261110.contracts import file_digest
from scripts.campaign_20261110.tests.test_runtime_suite import fixture,put


def second(root, *, start=2000):
    old,env,cfg=fixture(root)
    pointer=json.loads((root/'runtime_validation/current.json').read_text())
    new=root/'runtime_validation/run2';native=root/'code/frozen2';native.mkdir()
    (native/'.snapshot_commit').write_text('def\n')
    runtime={'path':str(native),'snapshot_sha256':file_digest(native/'.snapshot_commit'),'commit':'def'}
    release=json.loads(Path(pointer['release']['path']).read_text())
    release.update(run_id='run2',run_started_at=2,frozen_runtime=runtime)
    rr=put(new/'release.json',release)
    pointer2={'schema':pointer['schema'],'release':rr,
              'manifest':put(new/'manifest.json',json.loads(Path(pointer['manifest']['path']).read_text())),
              'index_reservation':put(new/'index_reservation.json',{'proof':{'start':start,'reserved_count':300}})}
    pr=put(new/'registered_pointer.json',pointer2)
    cfg2=copy.deepcopy(cfg);cfg2['validation_key']['code_sha256']=runtime['snapshot_sha256']
    cr=put(new/'config.json',cfg2);goal=put(new/'goal.json',{'group_id':'new0'})
    row=json.loads((old/'rows/class0.json').read_text())
    row.update(run_id='run2',run_started_at=2,release=rr,frozen_runtime=runtime,campaign_config=cr,
               goal_plan=goal,group_id='new0',fresh_start=start)
    rf=put(new/'rows/class0.json',row)
    env2={**env,'ISPATIALGYM_RUNTIME_SUITE_RUN':'run2','ISPATIALGYM_RUNTIME_SUITE_ROW':rf['path'],
          'ISPATIALGYM_RUNTIME_SUITE_ROW_SHA256':rf['sha256'],'ISPATIALGYM_CAMPAIGN_CONFIG':cr['path'],
          'ISPATIALGYM_CAMPAIGN_GOAL_PLAN':goal['path'],'ISGE_CODE':str(native)}
    approval=put(new/'approval.json',{'schema':'isg-runtime-suite-run-approval-v1','owner':'/root','approved':True,
        'pointer':pr,'required_validation_run':True,'production_authorization':False})
    return old,env,cfg,new,env2,cfg2,pr,approval


def test_registered_new_run_preserves_old_authorization_and_exact_binding(tmp_path):
    old,e,cfg,new,e2,cfg2,p,a=second(tmp_path)
    before=(tmp_path/'runtime_validation/current.json').read_bytes()
    assert not G.check_environment(tmp_path,e2,start=2000,count=3)['allowed']
    S.register_run(tmp_path,p,a,required_validation_run=True)
    assert (tmp_path/'runtime_validation/current.json').read_bytes()==before
    import stat
    assert stat.S_IMODE((tmp_path/'runtime_validation/authorized_runs.json').stat().st_mode)==0o640
    for env,start in [(e,1000),(e2,2000)]:
        for kind in ('collection','source_validation'):
            assert G.check_environment(tmp_path,env,start=start,count=3,kind=kind)['allowed']
    assert S.active(tmp_path)[4]==old
    assert S.active(tmp_path,run_dir=new)[4]==new
    assert S.active(tmp_path,run_id='run2')[4]==new
    assert P.class_rows(tmp_path,new,new/'rows/class0.json')==[new/'rows/class0.json']
    assert not G.check_environment(tmp_path,{**e2,'ISPATIALGYM_RUNTIME_SUITE_RUN':'run'},start=2000,count=3)['allowed']
    assert not G.check_environment(tmp_path,{**e2,'ISGE_CODE':e['ISGE_CODE']},start=2000,count=3)['allowed']
    assert not G.check_environment(tmp_path,e2,start=1000,count=3)['allowed']


def test_unregistered_directory_and_conflicting_run_selectors_fail(tmp_path):
    old,e,cfg,new,e2,cfg2,p,a=second(tmp_path)
    with pytest.raises(ValueError):S.active(tmp_path,run_dir=new)
    S.register_run(tmp_path,p,a,required_validation_run=True)
    with pytest.raises(ValueError):S.active(tmp_path,run_dir=old,run_id='run2')
    with pytest.raises(ValueError):S.active(tmp_path,run_dir=tmp_path/'out/run2')


def test_required_validation_cannot_borrow_100_old_run_results(tmp_path,monkeypatch):
    from scripts.campaign_20261110 import runtime_suite_acceptance as A
    old,e,cfg,new,e2,cfg2,p,a=second(tmp_path)
    for i in range(100):
        put(old/'acceptance'/f'class{i}.json',{'run_id':'run','class_id':f'class{i}','validation_passed':True})
        put(new/'acceptance'/f'class{i}.json',{'run_id':'run','class_id':f'class{i}','validation_passed':True})
    monkeypatch.setattr(A,'verify_result',lambda root,ref:json.loads(Path(ref['path']).read_text()))
    assert S.current_training_ready(tmp_path)
    S.register_run(tmp_path,p,a,required_validation_run=True)
    assert not S.current_training_ready(tmp_path)
    for i in range(100):put(new/'acceptance'/f'class{i}.json',{'run_id':'run2','class_id':f'class{i}','validation_passed':True})
    assert S.current_training_ready(tmp_path)
    put(new/'acceptance/class99.json',{'run_id':'run2','class_id':'class99','validation_passed':False})
    assert not S.current_training_ready(tmp_path)


@pytest.mark.parametrize('mutation',['approval','pointer','overlap','catalog'])
def test_registration_rejects_unapproved_or_incoherent_run(tmp_path,mutation):
    old,e,cfg,new,e2,cfg2,p,a=second(tmp_path,start=1100 if mutation=='overlap' else 2000)
    if mutation=='approval':
        v=json.loads(Path(a['path']).read_text());v['owner']='/root/other';a=put(Path(a['path']),v)
    if mutation=='pointer':Path(p['path']).write_text('{}')
    if mutation=='catalog':
        v=json.loads(Path(p['path']).read_text());manifest=json.loads(Path(v['manifest']['path']).read_text());manifest['classes'].reverse();v['manifest']=put(new/'manifest.json',manifest);p=put(Path(p['path']),v)
        ap=json.loads(Path(a['path']).read_text());ap['pointer']=p;a=put(Path(a['path']),ap)
    with pytest.raises(ValueError):S.register_run(tmp_path,p,a,required_validation_run=True)
    assert not (tmp_path/'runtime_validation/authorized_runs.json').exists()


def test_required_pointer_tampering_and_changed_native_fail_closed(tmp_path):
    old,e,cfg,new,e2,cfg2,p,a=second(tmp_path)
    S.register_run(tmp_path,p,a,required_validation_run=True)
    path=tmp_path/'runtime_validation/authorized_runs.json';v=json.loads(path.read_text());v['required_validation_run']['pointer']={'path':'wrong','sha256':'0'*64};put(path,v)
    assert not S.current_training_ready(tmp_path)
    Path(e2['ISGE_CODE'],'.snapshot_commit').write_text('different\n')
    assert not G.check_environment(tmp_path,e2,start=2000,count=3)['allowed']
    assert G.check_environment(tmp_path,e,start=1000,count=3)['allowed']


def test_missing_required_target_does_not_fall_back_to_old_run(tmp_path):
    old,e,cfg,new,e2,cfg2,p,a=second(tmp_path)
    S.register_run(tmp_path,p,a,required_validation_run=True)
    path=tmp_path/'runtime_validation/authorized_runs.json';v=json.loads(path.read_text())
    del v['required_validation_run'];put(path,v)
    with pytest.raises(ValueError,match='target is missing'):S.required_run(tmp_path)
    assert not S.current_training_ready(tmp_path)
    assert G.check_environment(tmp_path,e,start=1000,count=3)['allowed']
    assert G.check_environment(tmp_path,e2,start=2000,count=3)['allowed']


@pytest.mark.parametrize('private',['file','directory'])
def test_registration_requires_shared_control_closure(tmp_path,private):
    old,e,cfg,new,e2,cfg2,p,a=second(tmp_path)
    target=Path(a['path']) if private=='file' else new
    target.chmod(0o600 if private=='file' else 0o700)
    with pytest.raises(ValueError,match='shared-'):S.register_run(tmp_path,p,a,required_validation_run=True)
    assert not (tmp_path/'runtime_validation/authorized_runs.json').exists()


def test_observer_selects_registered_run_and_writes_only_its_status(tmp_path):
    from scripts.campaign_20261110 import runtime_suite_observer as O
    old,e,cfg,new,e2,cfg2,p,a=second(tmp_path)
    S.register_run(tmp_path,p,a,required_validation_run=True)
    put(old/'acceptance_observer.json',{'run_id':'run','classes':{'class0':{'validation_passed':True}}})
    before=(old/'acceptance_observer.json').read_bytes()
    result=O.observe(tmp_path,new)
    assert result['run_id']=='run2' and len(result['classes'])==100
    assert result['classes']['class0']['status']=='awaiting_allocation'
    assert json.loads((new/'acceptance_observer.json').read_text())==result
    assert (old/'acceptance_observer.json').read_bytes()==before
