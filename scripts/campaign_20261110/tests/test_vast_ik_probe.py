import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.campaign_20261110 import vast_ik_probe as probe
from scripts.campaign_20261110.tests.test_ik_native_probe import ProbeBoundaryTests

@pytest.fixture
def prepared():
    fixture=ProbeBoundaryTests();fixture.setUp()
    root=fixture.root
    config=Path(fixture.spec['config']);cfg=json.loads(config.read_text());cfg['template'].update(class_id='inter_object_relation_satisfaction-touching_04',scene='Rs_int');config.write_text(json.dumps(cfg));fixture.spec['inputs'][str(config)]=probe.sha(config)
    helper=root/'code/helper';scripts=helper/'scripts/campaign_20261110';scripts.mkdir(parents=True)
    for name in ('vast_ik_probe.py','ik_native_probe.py','job_ik_native_probe.sh'):(scripts/name).write_text(name)
    release=root/'release';release.mkdir();anchor=release/'anchors.json';anchor.write_text('{}');(release/'release_manifest.json').write_text(json.dumps({'files':{'anchors.json':probe.sha(anchor)}}))
    spec=root/'spec.json';spec.write_text(json.dumps(fixture.spec))
    env=probe.library_environment(root,release);env.update(ISPATIALGYM_CAMPAIGN_CONFIG=str(config),ISPATIALGYM_EXACT_IK_MANIFEST=fixture.spec['manifest'])
    yield root,spec,helper,release,env
    fixture.tearDown()

def make(data):
    root,spec,helper,release,env=data
    return probe.prepare(root,spec,helper,env,'bounded_probe',release)

def test_typed_probe_never_activates_or_changes_source(prepared):
    root=prepared[0];before={str(p):p.read_bytes()for p in root.rglob('*')if p.is_file()};job=make(prepared)
    assert all(Path(p).read_bytes()==b for p,b in before.items())
    jid=probe.enqueue(job,root);q=json.loads((root/'vast/queue'/f'{jid}.json').read_text())
    assert q['diagnostic_only'] and q['benchmark_credit']==0 and q['exact_ik_exchange_required']
    assert not(root/'vast/config.json').exists()
    with pytest.raises(ValueError,match='already enqueued'):probe.enqueue(job,root)

@pytest.mark.parametrize('changed',['customs','manifest','snapshot'])
def test_changed_physics_or_compatibility_blocks_queue(prepared,changed):
    job=make(prepared);s=json.loads(prepared[1].read_text());p=Path(s[changed]);p=p/'.snapshot_commit'if changed=='snapshot'else p;p.write_text('changed')
    with pytest.raises(ValueError,match='bytes changed'):probe.enqueue(job,prepared[0])
    assert not(prepared[0]/'vast/queue').exists()

def test_changed_library_payload_blocks_queue(prepared):
    job=make(prepared);(prepared[3]/'anchors.json').write_text('changed')
    with pytest.raises(ValueError,match='bytes changed'):probe.enqueue(job,prepared[0])

@pytest.mark.parametrize('published,exit_code,expected',[(False,0,0),(True,0,2),(False,1,2)])
def test_report_requires_no_real_instance_publication(prepared,monkeypatch,published,exit_code,expected):
    job=make(prepared);value=probe.verify(job);out=Path(value['probe_output'])
    for k in prepared[4]:monkeypatch.setenv(k,'old')
    monkeypatch.setenv('ISPATIALGYM_EXACT_IK_LOCAL_CACHE','worker-owned-cache')
    def run(*args,**kwargs):
        assert probe.os.environ['ISPATIALGYM_EXACT_IK_LOCAL_CACHE']=='worker-owned-cache'
        out.mkdir(parents=True);(out/'probe_report.json').write_text(json.dumps({'published_tro_files':0,'status':'one_draw_finished','sampler_exit_code':exit_code}))
        if published:(out/'forbidden-tro_state.json').write_text('{}')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(probe.subprocess,'run',run)
    assert probe.run(job)==expected
    completion=json.loads((job/'probe_completion.json').read_text());assert completion['benchmark_credit']==0
    assert completion['actual_published_tro_files']==int(published)
    assert (job/'result/probe_report.json').exists()
