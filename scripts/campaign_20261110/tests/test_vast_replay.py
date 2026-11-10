import json
from pathlib import Path

import h5py
import pytest

from scripts.campaign_20261110 import vast_replay as vr


def fixture(root):
    source=root/'out/source'; template=source/'inst/task/scene'; template.mkdir(parents=True)
    runtime=root/'code/native'; helper=root/'code/helper'; runtime.mkdir(parents=True)
    (runtime/'.snapshot_commit').write_text('native unchanged')
    for name in ('job_replay.sh','infra_open_loop.sh','infra_replay_scene.py','vast_replay.py'):
        path=helper/'scripts/campaign_20261110'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(name)
    objects={'objects_info':{'init_info':{'robot':{},'target':{}}}}
    (template/'template.json').write_text(json.dumps(objects));(template/'template_meta.json').write_text('{}')
    instances=template/'instances';instances.mkdir();(instances/'i-tro_state.json').write_text('{}')
    demo=source/'demo/i/demo.hdf5';demo.parent.mkdir(parents=True)
    with h5py.File(demo,'w') as stream:
        data=stream.create_group('data');data.attrs['scene_file']=json.dumps(objects)
    (demo.parent/'meta.json').write_text(json.dumps({'success':True,'template_dir':str(template),
        'provenance':{'isg_root':str(runtime)},'load_rooms':{'rooms':[]}}))
    (source/'job.env').write_text('ISGE_CODE='+str(runtime)+'\n')
    (source/'sampling.done').touch()
    return source,runtime,helper,root/'validation/canary'


def test_transport_preserves_source_and_queue_is_not_activated(tmp_path):
    source,runtime,helper,out=fixture(tmp_path)
    before={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    job=vr.prepare(tmp_path,source,runtime,helper,out,'first')
    assert before=={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    assert not out.exists() and not (tmp_path/'vast/queue').exists()
    jid=vr.enqueue(job,tmp_path)
    assert jid==vr.enqueue(job,tmp_path)
    record=json.loads((tmp_path/'vast/queue'/f'{jid}.json').read_text())
    assert record['job_kind']=='physics_replay' and record['source_job']==str(source)
    assert not (tmp_path/'vast/config.json').exists()
    assert not json.loads((job/'vast_submission.json').read_text())['activated']


def test_changed_source_or_runtime_is_rejected_before_activation(tmp_path):
    source,runtime,helper,out=fixture(tmp_path)
    job=vr.prepare(tmp_path,source,runtime,helper,out,'first')
    (source/'demo/i/demo.hdf5').write_bytes(b'changed')
    with pytest.raises(ValueError,match='source bytes changed'):vr.enqueue(job,tmp_path)
    assert not (tmp_path/'vast/queue').exists()


def test_unclosed_source_and_different_native_runtime_fail_closed(tmp_path):
    source,runtime,helper,out=fixture(tmp_path);(source/'sampling.done').unlink()
    with pytest.raises(ValueError,match='not closed'):vr.prepare(tmp_path,source,runtime,helper,out,'first')
    (source/'sampling.done').touch()
    with pytest.raises(ValueError,match='source runtime differs'):vr.prepare(tmp_path,source,helper,helper,out,'first')


def test_binding_preserves_prior_gate_evidence_and_keeps_native_paths_inside_attempt(tmp_path):
    source,runtime,helper,out=fixture(tmp_path)
    job=vr.prepare(tmp_path,source,runtime,helper,out,'first')
    out.mkdir(parents=True);(out/'gate.json').write_text('existing failed replay')
    with pytest.raises(ValueError,match='existing evidence'):vr.bind_output(job,tmp_path)
    assert (out/'gate.json').read_text()=='existing failed replay'
    (out/'gate.json').unlink();vr.bind_output(job,tmp_path);vr.bind_output(job,tmp_path)
    (job/'gate/replay/i').mkdir(parents=True)
    (job/'gate/gate.json').write_text('{}');(job/'gate/replay/i/meta.json').write_text('{}')
    assert (out/'gate.json').resolve().parent==(job/'gate').resolve()
    assert (out/'replay/i/meta.json').resolve().is_relative_to((out/'gate.json').resolve().parent)


def test_transport_environment_and_queue_record_are_pinned(tmp_path):
    source,runtime,helper,out=fixture(tmp_path)
    job=vr.prepare(tmp_path,source,runtime,helper,out,'first');jid=vr.enqueue(job,tmp_path)
    record=tmp_path/'vast/queue'/f'{jid}.json';record.write_text('{}')
    with pytest.raises(ValueError,match='ambiguous replay submission'):vr.enqueue(job,tmp_path)
    (job/'job.env').write_text('ISGE_CODE=/different/runtime\n')
    with pytest.raises(ValueError,match='source bytes changed'):vr.verify(job,tmp_path)
