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


def consumed_fixture(root, constraint='attachment_free', metadata_constraint=None, rest_hint=None):
    plan,native,helper,library=fixture(root)
    base=plan.parent; old=json.loads(plan.read_text()); unit=old['units'][0]; entry=unit['job']
    entry.update(kind='attachment' if constraint == 'attachment_free' else 'tabletop',constraint=constraint)
    if rest_hint is not None: entry['rest_hint_wxyz'] = rest_hint
    value = json.loads(library.read_text())
    value.update(category=entry['category'], model=entry['model'], scale=entry['scale'], constraint=metadata_constraint or constraint)
    v.write(library, value)
    selected=base/'jobs.json';v.write(selected,[entry]);anchors=base/'anchors.json'
    identity={'kind':'tabletop','library_identity':'test-native-identity','library_sha256':v.sha(library),'height':.8126,'arms':['right']}
    key=v.hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    ref=lambda p:{'path':str(p),'sha256':v.sha(p)}
    inputs=[ref(p)for p in [library,library.parent/'geometry.json',selected,anchors,native/'.snapshot_commit']]
    argv=['python','-u','-m','ispatialgym.sampling.reach_maps','compute','--kind','tabletop','--jobs',str(selected),
          '--glib',str(library.parents[2]),'--anchors',str(anchors),'--heights','.8126','--arm','right','--shard','0/1','--out',str(base/'maps'),
          '--mode','full','--lattice','0.08','--seeds','8','--trunk-eligibility','per_d_e']
    prepared=base/'prepared.json';v.write(prepared,{'physical_dedup_key':key,'physical_identity':identity,'native_runtime':str(native),
           'native_argv_by_arm':{'right':argv},'height':'.8126','configuration_ids':['selected-config']})
    ticket={'computation_kind':'reach_map','consumer_kind':'tabletop','physical_dedup_key':key,'dedup_key':key,
            'native_time_seconds':1500,'hard_time_seconds':1800,'benchmark_credit':0,'production_authorized':False,
            'input_refs':inputs,'scripts':[],'prepared':ref(prepared),'runtime_ref':ref(native/'.snapshot_commit'),
            'configuration_ids':['selected-config'],'expected_account':'minyeong'}
    for field,name in [('claim_path','claims'),('submission_path','receipts')]:
        ticket[field]=str(root/'infra/coordination/preparation_dispatch/consumed_asset_builds_r1'/name/(key+'.json'))
    tickets=base/'tickets.json';v.write(tickets,{'schema':'isg-preparation-dispatch-tickets-v1','jobs':[ticket]})
    return tickets,native,helper,library,ticket


def test_consumed_attachment_map_preserves_physical_claim_and_finite_bounds(tmp_path):
    tickets,native,helper,library,ticket=consumed_fixture(tmp_path)
    job=v.prepare_consumed(tmp_path,tickets,0,native,helper,'consumed');value=v.verify(job,tmp_path)
    assert value['physical_dedup_key']==ticket['dedup_key']and value['claim_path']==ticket['claim_path']
    assert value['unit']['arms']==['right']and value['unit']['job']['constraint']=='attachment_free'
    assert value['time_limit_s']==1500 and value['scenes']==[]
    jid=v.enqueue(job,tmp_path);queued=json.loads((tmp_path/'vast/queue'/(jid+'.json')).read_text())
    assert queued['time_s']==1800 and queued['physical_dedup_key']==ticket['dedup_key']
    assert not Path(ticket['claim_path']).exists() and not(tmp_path/'vast/config.json').exists()


@pytest.mark.parametrize('constraint', ['tabletop_insert', 'flower_stem_top'])
def test_consumed_specialized_map_keeps_exact_constraint(tmp_path, constraint):
    tickets,native,helper,library,ticket=consumed_fixture(tmp_path, constraint)
    job=v.prepare_consumed(tmp_path,tickets,0,native,helper,'specialized')
    value=v.verify(job,tmp_path)
    assert value['unit']['job']['constraint'] == constraint
    assert value['physical_dedup_key'] == ticket['physical_dedup_key']
    assert not(tmp_path/'vast/queue').exists()


@pytest.mark.parametrize('change', ['wrong_constraint', 'unreviewed_constraint', 'designated_rest'])
def test_consumed_specialized_map_rejects_wrong_binding(tmp_path, change):
    kwargs = {'constraint': 'tabletop_insert'}
    if change == 'wrong_constraint': kwargs['metadata_constraint'] = 'tabletop'
    if change == 'unreviewed_constraint': kwargs['constraint'] = 'floating'
    if change == 'designated_rest': kwargs['rest_hint'] = [1,0,0,0]
    tickets,native,helper,library,ticket=consumed_fixture(tmp_path, **kwargs)
    with pytest.raises(ValueError): v.prepare_consumed(tmp_path,tickets,0,native,helper,'wrong')
    assert not(tmp_path/'vast/queue').exists()


@pytest.mark.parametrize('change',['scene_command','altered_height','unbound_library','missing_payload','changed_claim'])
def test_consumed_map_rejects_broadened_or_changed_recipe(tmp_path,change):
    tickets,native,helper,library,ticket=consumed_fixture(tmp_path)
    prepared=Path(ticket['prepared']['path']);value=json.loads(prepared.read_text())
    if change=='scene_command':value['native_argv_by_arm']['right'][3]='ispatialgym.sampling.sample_isg'
    elif change=='altered_height':value['native_argv_by_arm']['right'][14]='.9'
    elif change=='unbound_library':ticket['input_refs']=[r for r in ticket['input_refs']if r['path']!=str(library)]
    elif change=='missing_payload':library.unlink()
    else:ticket['claim_path']=str(tmp_path/'unrelated/claim.json')
    v.write(prepared,value);ticket['prepared']['sha256']=v.sha(prepared)
    v.write(tickets,{'schema':'isg-preparation-dispatch-tickets-v1','jobs':[ticket]})
    with pytest.raises((ValueError,OSError)):v.prepare_consumed(tmp_path,tickets,0,native,helper,'bad')
    assert not(tmp_path/'vast/queue').exists()
