import json
from pathlib import Path

import pytest

from scripts.campaign_20261110 import vast_native_dependency as native


def fixture(root,monkeypatch):
    runtime=root/'code/native';runtime.mkdir(parents=True);(runtime/'.snapshot_commit').write_text('same native runtime')
    helper=root/'code/helper/scripts/campaign_20261110';helper.mkdir(parents=True)
    for name in ('vast_native_dependency.py','job_native_dependency.sh'):(helper/name).write_text(name)
    wrapper=root/'tools/original.sh';wrapper.parent.mkdir();wrapper.write_text('original native wrapper')
    source=root/'candidate';source.mkdir();bddl=source/'bddl';bddl.mkdir();(bddl/'problem.bddl').write_text('native problem')
    handles=root/'proofs/handles.json';handles.parent.mkdir();handles.write_text('{"native_handle":true}')
    customs=source/'customs.json';customs.write_text(json.dumps({'handle_file':str(handles)}))
    scenes=root/'scenes';scenes.mkdir();(scenes/'scene.json').write_text('{}')
    anchor=root/'shared/anchors.json';anchor.parent.mkdir();anchor.write_text('{}')
    monkeypatch.setattr(native,'BASE_ENV',{'ISPATIALGYM_REACH_MAP_ANCHORS':str(anchor),'ISPATIALGYM_V2_ONLY':'1'})
    entry={'production_released':False,'class_id':'wardrobe_open','scene':'office_cubicles_right',
        'environment':{'ISPATIALGYM_STABLE_SCENE_DIR':str(scenes)},
        'sample_argv':['python','-m','ispatialgym.sampling.sample_isg','--variant','wardrobe_open','--customs',str(customs),
            '--bddl-root',str(bddl),'--no-write-back','--out','${CAMPAIGN_OUT}/validation/left']}
    manifest=source/'validation_jobs.json';manifest.write_text(json.dumps({'jobs':[entry]}))
    output=root/'materialized_runs/wardrobe/validation/left'
    return manifest,runtime,helper.parents[1],wrapper,output


def test_native_source_and_reserved_index_are_preserved_and_queue_stays_inactive(tmp_path,monkeypatch):
    manifest,runtime,helper,wrapper,out=fixture(tmp_path,monkeypatch)
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    job=native.prepare(tmp_path,manifest,0,runtime,helper,wrapper,out,50108200,'left')
    assert all(Path(name).read_bytes()==body for name,body in before.items())
    value=native.verify(job,tmp_path)
    assert (job/'sample.log').is_symlink() and (job/'sample.log').readlink()==Path('result/sample.log')
    assert value['argv'][-4:]==['--start-index','50108200','--time-budget-s','1800']
    assert str(tmp_path/'proofs/handles.json') in value['files']
    jid=native.enqueue(job,tmp_path);assert native.enqueue(job,tmp_path)==jid
    record=json.loads((tmp_path/'vast/queue'/f'{jid}.json').read_text())
    assert record['job_kind']=='native_dependency' and not (tmp_path/'vast/config.json').exists()
    assert not out.exists()


def test_changed_native_handle_or_runtime_prevents_dispatch(tmp_path,monkeypatch):
    manifest,runtime,helper,wrapper,out=fixture(tmp_path,monkeypatch)
    job=native.prepare(tmp_path,manifest,0,runtime,helper,wrapper,out,50108200,'left')
    (tmp_path/'proofs/handles.json').write_text('changed geometry')
    with pytest.raises(ValueError,match='bytes changed'):native.enqueue(job,tmp_path)
    assert not (tmp_path/'vast/queue').exists()


def test_private_native_interior_registry_is_pinned_and_changes_block_dispatch(tmp_path,monkeypatch):
    manifest,runtime,helper,wrapper,out=fixture(tmp_path,monkeypatch)
    registry=tmp_path/'proofs/interiors.json'
    registry.write_text('{"interiors":{"measured_bookcase":{"height":0.8354}}}')
    entries=json.loads(manifest.read_text())
    entries['jobs'][0]['environment']['ISPATIALGYM_INTERIORS']=str(registry)
    manifest.write_text(json.dumps(entries))
    job=native.prepare(tmp_path,manifest,0,runtime,helper,wrapper,out,50108200,'left')
    value=native.verify(job,tmp_path)
    assert value['environment']['ISPATIALGYM_INTERIORS']==str(registry)
    assert value['files'][str(registry)]==native.sha(registry)
    registry.write_text('{"changed_geometry":true}')
    with pytest.raises(ValueError,match='bytes changed'):native.enqueue(job,tmp_path)


def test_native_interior_registry_cannot_escape_campaign(tmp_path,monkeypatch):
    manifest,runtime,helper,wrapper,out=fixture(tmp_path,monkeypatch)
    entries=json.loads(manifest.read_text())
    entries['jobs'][0]['environment']['ISPATIALGYM_INTERIORS']=str(tmp_path/'proofs')
    manifest.write_text(json.dumps(entries))
    with pytest.raises(ValueError,match='explicit JSON file'):
        native.prepare(tmp_path,manifest,0,runtime,helper,wrapper,out,50108200,'left')


@pytest.mark.parametrize('class_id,option,valid', [
    ('inter_object_relation_satisfaction-ontop_large_scale_01','1',True),
    ('wardrobe_open','1',False),
    ('inter_object_relation_satisfaction-ontop_large_scale_01','off',False),
])
def test_exact_soap_screen_is_pinned_without_becoming_a_file_dependency(tmp_path,monkeypatch,class_id,option,valid):
    manifest,runtime,helper,wrapper,out=fixture(tmp_path,monkeypatch)
    entries=json.loads(manifest.read_text())
    entries['jobs'][0]['class_id']=class_id
    flag='ISPATIALGYM_CAMPAIGN_SOAP_LIFT_FRAME_SCREEN'
    entries['jobs'][0]['environment'][flag]=option
    manifest.write_text(json.dumps(entries))
    if not valid:
        with pytest.raises(ValueError,match='audited exact soap class'):
            native.prepare(tmp_path,manifest,0,runtime,helper,wrapper,out,50108200,'left')
        return
    job=native.prepare(tmp_path,manifest,0,runtime,helper,wrapper,out,50108200,'left')
    value=native.verify(job,tmp_path)
    assert value['environment'][flag]=='1'
    assert str(manifest) in value['files']
    entries['jobs'][0]['environment'][flag]='off'
    manifest.write_text(json.dumps(entries))
    with pytest.raises(ValueError,match='bytes changed'):
        native.enqueue(job,tmp_path)


def test_existing_native_output_or_wrong_entrypoint_cannot_be_replaced(tmp_path,monkeypatch):
    manifest,runtime,helper,wrapper,out=fixture(tmp_path,monkeypatch)
    out.mkdir(parents=True);(out/'sample.log').write_text('native started')
    with pytest.raises(ValueError,match='already has evidence'):
        native.prepare(tmp_path,manifest,0,runtime,helper,wrapper,out,50108200,'left')
    (out/'sample.log').unlink();out.rmdir()
    value=json.loads(manifest.read_text());value['jobs'][0]['sample_argv'][2]='unapproved.module';manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='unsupported native'):
        native.prepare(tmp_path,manifest,0,runtime,helper,wrapper,out,50108200,'left')


def test_native_completion_requires_real_tro_and_retains_failed_sampler_result(tmp_path,monkeypatch):
    manifest,runtime,helper,wrapper,out=fixture(tmp_path,monkeypatch)
    job=native.prepare(tmp_path,manifest,0,runtime,helper,wrapper,out,50108200,'left')
    for key in native.verify(job)['environment']:
        monkeypatch.setenv(key,native.os.environ.get(key,''))
    class Result:returncode=0
    monkeypatch.setattr(native.subprocess,'run',lambda *args,**kwargs:Result())
    assert native.run(job)==2  # A zero process exit without a native instance is insufficient.
    result=json.loads((job/'result/sampling_gate.json').read_text())
    assert result['tro_count']==0 and result['validation_only'] is True


def test_exact_campaign_library_release_and_materialized_placeholder_are_preserved(tmp_path,monkeypatch):
    manifest,runtime,helper,wrapper,out=fixture(tmp_path,monkeypatch)
    entry=json.loads(manifest.read_text());entry['jobs'][0]['sample_argv'][6]='${MATERIALIZED}/customs.json'
    manifest.write_text(json.dumps(entry))
    release=tmp_path/'release';(release/'glib').mkdir(parents=True);(release/'maps').mkdir()
    (release/'anchors.json').write_text('{"native":"anchors"}')
    library=release/'glib/library.json';library.write_text('{"native":"grasps"}')
    release_manifest=release/'release_manifest.json'
    release_manifest.write_text(json.dumps({'files':{str(p.relative_to(release)):native.sha(p)
        for p in [release/'anchors.json',library]}}))
    job=native.prepare(tmp_path,manifest,0,runtime,helper,wrapper,out,50108200,'left',
                       v2_root=release,v2_manifest_sha256=native.sha(release_manifest))
    value=native.verify(job,tmp_path)
    assert value['argv'][6]==str(manifest.parent/'customs.json')
    assert value['environment']['ISPATIALGYM_GRASP_LIBRARY_V2']==str(release/'glib')
    library.write_text('changed native grasps')
    with pytest.raises(ValueError,match='library bytes changed'):native.enqueue(job,tmp_path)


def test_declared_short_canary_bounds_survive_source_transport_queue_and_process(tmp_path,monkeypatch):
    manifest,runtime,helper,wrapper,out=fixture(tmp_path,monkeypatch)
    source=json.loads(manifest.read_text());source['jobs'][0]['sample_argv']+=['--time-budget-s','1200']
    manifest.write_text(json.dumps(source));before=manifest.read_bytes()
    job=native.prepare(tmp_path,manifest,0,runtime,helper,wrapper,out,50108200,'bounded',
                       sampling_budget_s=1200,subprocess_timeout_s=1800)
    assert manifest.read_bytes()==before
    value=native.verify(job,tmp_path)
    assert value['runtime_bounds']=={'sampling_budget_s':1200,'subprocess_timeout_s':1800,'allocation_seconds':2400}
    assert value['argv'].count('--time-budget-s')==1
    jid=native.enqueue(job,tmp_path)
    assert json.loads((tmp_path/'vast/queue'/f'{jid}.json').read_text())['time_s']==2400
    for key in value['environment']:monkeypatch.setenv(key,native.os.environ.get(key,''))
    def run(argv,**kwargs):
        assert kwargs['timeout']==1800 and argv[argv.index('--time-budget-s')+1]=='1200'
        raise native.subprocess.TimeoutExpired(argv,1800)
    monkeypatch.setattr(native.subprocess,'run',run)
    assert native.run(job)==2
    assert json.loads((job/'result/sampling_gate.json').read_text())['returncode']==124


@pytest.mark.parametrize('sampling,timeout', [(0,1800),(1801,2400),(1200,1200),(1200,2401),(True,1800),(1200,float('inf'))])
def test_invalid_or_expanded_bounds_rejected_before_preparation(tmp_path,monkeypatch,sampling,timeout):
    manifest,runtime,helper,wrapper,out=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError):
        native.prepare(tmp_path,manifest,0,runtime,helper,wrapper,out,50108200,'bad',
                       sampling_budget_s=sampling,subprocess_timeout_s=timeout)
    assert not (tmp_path/'out').exists()


def test_conflicting_source_budget_is_not_silently_overwritten(tmp_path,monkeypatch):
    manifest,runtime,helper,wrapper,out=fixture(tmp_path,monkeypatch)
    source=json.loads(manifest.read_text());source['jobs'][0]['sample_argv']+=['--time-budget-s','1200']
    manifest.write_text(json.dumps(source))
    with pytest.raises(ValueError,match='conflicts'):
        native.prepare(tmp_path,manifest,0,runtime,helper,wrapper,out,50108200,'bad')
    assert not (tmp_path/'out').exists()


def test_legacy_inputs_keep_the_original_bounds_and_changed_argv_is_rejected(tmp_path,monkeypatch):
    manifest,runtime,helper,wrapper,out=fixture(tmp_path,monkeypatch)
    job=native.prepare(tmp_path,manifest,0,runtime,helper,wrapper,out,50108200,'legacy')
    path=job/'native_inputs.json';value=json.loads(path.read_text());value.pop('runtime_bounds')
    path.write_text(json.dumps(value));assert native.verified_bounds(native.verify(job))['subprocess_timeout_s']==2400
    value['argv'][value['argv'].index('--time-budget-s')+1]='3600';path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='sampler budget'):native.verify(job)


@pytest.mark.parametrize('class_id,value,allowed', [
    ('camera_relocalization-obstructed-hamper_1','1',True),
    ('camera_relocalization-obstructed-hamper_1','0',False),
    ('object_scaling-apple','1',False)])
def test_stationary_camera_goal_is_exactly_scoped(tmp_path,monkeypatch,class_id,value,allowed):
    manifest,runtime,helper,wrapper,out=fixture(tmp_path,monkeypatch)
    source=json.loads(manifest.read_text());source['jobs'][0]['class_id']=class_id
    source['jobs'][0]['environment']['ISPATIALGYM_CAMPAIGN_OBSTACLE_GOAL']=value
    manifest.write_text(json.dumps(source))
    if not allowed:
        with pytest.raises(ValueError,match='exact obstructed camera'):
            native.prepare(tmp_path,manifest,0,runtime,helper,wrapper,out,600,'camera')
        return
    job=native.prepare(tmp_path,manifest,0,runtime,helper,wrapper,out,600,'camera')
    assert native.verify(job)['environment']['ISPATIALGYM_CAMPAIGN_OBSTACLE_GOAL']=='1'
