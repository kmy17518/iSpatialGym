import json
from pathlib import Path
import pytest
from scripts.campaign_20261110 import vast_map_prebuild as v


def fixture(root):
    native=root/'code/native'; helper=root/'code/helper'
    for p in [native/'.snapshot_commit',native/'scripts/grasp_bands/job_glib.sh',helper/'scripts/campaign_20261110/vast_prebuild.py',helper/'scripts/campaign_20261110/vast_map_prebuild.py',helper/'scripts/campaign_20261110/job_map_prebuild.sh']:
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text('frozen exact code')
    base=root/'training_preparation/maps';base.mkdir(parents=True)
    library=base/'glib/models/m/library.json';v.write(library,{'grasps':[{'rest_id':'r0'}]});v.write(library.parent/'geometry.json',{'vertices':[1,2,3]})
    for name in ['probe.json','scene.json']:v.write(base/name,{'measured':True})
    row={'id':'m','ready':True,'kind':'tabletop','constraint':'tabletop','category':'box','model':'m','scale':[1]*3}
    ref=lambda p:{'path':str(p),'sha256':v.sha(p)}
    req={'entry_id':'m','actual_height_key':'.8126','library':ref(library),'probe':ref(base/'probe.json'),'stable_scene':ref(base/'scene.json')}
    dep=base/'dependencies.json';v.write(dep,{'requests':[req]});anchors=base/'anchors.json';v.write(anchors,{'anchors':[{'key':'.8126','height':.8126}]})
    unit={'id':0,'job':row,'arms':['right','left'],'height':'.8126','source_dependencies':ref(dep),'source_library':ref(library),'anchors':str(anchors)}
    plan=base/'plan.json';v.write(plan,{'runtime':str(native),'units':[unit]});return plan,native,helper,library


def test_exact_map_unit_copy_and_idempotent_inactive_queue(tmp_path):
    plan,native,helper,library=fixture(tmp_path);before=library.read_bytes();job=v.prepare(tmp_path,plan,0,native,helper,'one');value=v.verify(job,tmp_path)
    assert library.read_bytes()==before
    assert Path(value['private_glib'],'models/m/library.json').read_bytes()==before
    assert value['unit']['height']=='.8126' and value['new_native_instances']==0
    jid=v.enqueue(job,tmp_path);assert v.enqueue(job,tmp_path)==jid
    assert not(tmp_path/'vast/config.json').exists()


def test_changed_library_fails_closed(tmp_path):
    plan,native,helper,library=fixture(tmp_path);job=v.prepare(tmp_path,plan,0,native,helper,'one');value=v.verify(job,tmp_path)
    Path(value['private_glib'],'models/m/library.json').write_text('changed')
    with pytest.raises(ValueError,match='input changed'):v.verify(job,tmp_path)
    assert not(job/'result').exists()


def test_unmeasured_height_and_checkpoint_payload_rejected(tmp_path):
    plan,native,helper,library=fixture(tmp_path);doc=json.loads(plan.read_text());doc['units'][0]['height']='.9000';v.write(plan,doc)
    with pytest.raises(ValueError,match='measured source missing'):v.prepare(tmp_path,plan,0,native,helper,'badheight')
    doc['units'][0]['height']='.8126';v.write(plan,doc);(library.parent/'weights.pt').write_text('not a geometry dependency')
    with pytest.raises(ValueError,match='only native geometry'):v.prepare(tmp_path,plan,0,native,helper,'weights')
    assert not(tmp_path/'vast/queue').exists()


def test_zero_exit_without_native_maps_is_failure(tmp_path,monkeypatch):
    plan,native,helper,library=fixture(tmp_path);job=v.prepare(tmp_path,plan,0,native,helper,'one')
    class Result:returncode=0
    calls=[]
    def run(argv,**kwargs):calls.append(argv);return Result()
    monkeypatch.setattr(v.subprocess,'run',run);assert v.run(job)==2
    assert len(calls)==2 and all('--skip-existing' in a for a in calls)
    report=json.loads((job/'result/map_prebuild_result.json').read_text());assert not report['map_files'] and report['positive_native_reader_required']


def secondary_scene(root, monkeypatch, destination=None):
    plan,native,helper,library=fixture(root)
    secondary=root.parent/'secondary/training_preparation'
    monkeypatch.setattr(v,'SECONDARY_PROOF_ROOT',secondary)
    scene=destination or secondary/'support/stable_scene.json'
    v.write(scene,{'measured':True,'exact_scene':'original'})
    doc=json.loads(plan.read_text());dep=Path(doc['units'][0]['source_dependencies']['path'])
    requests=json.loads(dep.read_text());requests['requests'][0]['stable_scene']={'path':str(scene),'sha256':v.sha(scene)}
    v.write(dep,requests);doc['units'][0]['source_dependencies']['sha256']=v.sha(dep);v.write(plan,doc)
    return plan,native,helper,scene


def test_secondary_measured_proof_is_exact_private_copy_and_worker_is_independent(tmp_path,monkeypatch):
    root=tmp_path/'primary';plan,native,helper,scene=secondary_scene(root,monkeypatch)
    before=scene.read_bytes();mode=scene.stat().st_mode
    job=v.prepare(root,plan,0,native,helper,'one');value=v.verify(job,root)
    copied=Path(value['proof_relocations'][str(scene)]['copy'])
    assert copied.read_bytes()==before and scene.read_bytes()==before and scene.stat().st_mode==mode
    assert str(scene) not in value['files'] and copied.is_relative_to(job/'proof_inputs')
    scene.unlink()  # Workers receive the bound copy, not cross-account source access.
    assert v.verify(job,root)['unit']['height']=='.8126'
    copied.write_text('changed')
    with pytest.raises(ValueError,match='input changed'):v.verify(job,root)


@pytest.mark.parametrize('mutation',['wrong_hash','wrong_source','extra_copy','unbound_copy'])
def test_proof_relocation_must_match_exact_measured_request(tmp_path,monkeypatch,mutation):
    root=tmp_path/'primary';plan,native,helper,scene=secondary_scene(root,monkeypatch)
    job=v.prepare(root,plan,0,native,helper,'one');path=job/'native_inputs.json';value=json.loads(path.read_text())
    ref=value['proof_relocations'][str(scene)]
    if mutation=='wrong_hash':ref['sha256']='0'*64
    elif mutation=='wrong_source':value['proof_relocations'][str(scene)+'wrong']=value['proof_relocations'].pop(str(scene))
    elif mutation=='extra_copy':value['proof_relocations']['/unrelated/private.json']=dict(ref)
    else:value['files'].pop(ref['copy'])
    v.write(path,value)
    with pytest.raises(ValueError,match='proof relocation'):v.verify(job,root)


def test_unrelated_external_and_symlink_proofs_are_rejected(tmp_path,monkeypatch):
    root=tmp_path/'primary';outside=tmp_path/'unrelated/scene.json'
    plan,native,helper,scene=secondary_scene(root,monkeypatch,outside)
    with pytest.raises(ValueError,match='outside approved'):v.prepare(root,plan,0,native,helper,'outside')
    allowed=v.SECONDARY_PROOF_ROOT/'support/scene.json';allowed.parent.mkdir(parents=True,exist_ok=True);allowed.symlink_to(outside)
    doc=json.loads(plan.read_text());dep=Path(doc['units'][0]['source_dependencies']['path']);requests=json.loads(dep.read_text())
    requests['requests'][0]['stable_scene']['path']=str(allowed);v.write(dep,requests);doc['units'][0]['source_dependencies']['sha256']=v.sha(dep);v.write(plan,doc)
    with pytest.raises(ValueError,match='outside approved'):v.prepare(root,plan,0,native,helper,'symlink')
