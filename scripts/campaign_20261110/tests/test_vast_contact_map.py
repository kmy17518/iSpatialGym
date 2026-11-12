import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.campaign_20261110 import vast_contact_map as v


def fixture(tmp_path):
    root = tmp_path/'campaign_20261110'
    native, helper = root/'code/native', root/'code/helper'
    for p in (native/'.snapshot_commit', helper/'scripts/campaign_20261110/vast_contact_map.py',
              helper/'scripts/campaign_20261110/vast_prebuild.py', helper/'scripts/campaign_20261110/job_contact_map.sh'):
        p.parent.mkdir(parents=True, exist_ok=True); p.write_text('frozen')
    base = root/'training_preparation/contact'; model = base/'glib/models/book__m__s1.1'
    for name in ('library.json', 'geometry.json', 'geometry.obj', 'rests.json'):
        v.write(model/name, {'original':name})
    obj = {'source':str(model), 'entry':model.name, 'scale':[1.1]*3, 'rest_id':'r0', 'geometry':'exact'}
    lib = base/'brace/models'/model.name/'library.json'; v.write(lib, {'object':obj, 'entries':[{'pose':'original'}]})
    audit, manifest = base/'audit.json', base/'manifest.json'
    v.write(audit, {'positive':True}); v.write(manifest, {'campaign_id':root.name,
        'configurations':[{'id':'configuration_one','class_id':'alignment-axial_02'}]})
    ref = lambda p: {'path':str(p),'sha256':v.sha(p)}
    unit = {'id':0,'native_runtime':str(native),'state':'ready_exact_map_dependency','positive_entries':1,
            'production_authorization':False,'configuration_validated':False,'arm':'left',
            'expected_map_config':{'arm':'left','lattice_m':.08,'seeds':8},'height':.5734,'height_key':'0.5734',
            'pose_set':{'digest':'exact-pose'},'entry_id':model.name,'map_kind':'axial_brace','library_kind':'brace',
            'source_files':{str(p):v.sha(p) for p in model.iterdir()},'source_library':ref(lib),
            'source_physics_audit':ref(audit),'selected_manifest':ref(manifest),'object':obj,'model_dir':str(model),
            'consumers':['configuration_one']}
    request = base/'request.json'
    v.write(request, {'campaign_id':root.name,'production_authorization':False,'native_runtime':str(native),'units':[unit]})
    return root, request, native, helper, lib


def prepared(tmp_path):
    root, request, native, helper, lib = fixture(tmp_path)
    job = v.prepare(root, request, 0, native, helper, 'one')
    return root, job, lib


def owner_and_capacity(root, job):
    value = v.verify(job); d = root/'infra/capacity_recovery'
    evidence = d/'retired.json'; v.write(evidence, {'retired_unit_ids':[0], 'campaign_id':root.name, 'source_plan':{'path':value['source_plan'],'sha256':v.sha(value['source_plan'])}})
    ref = {'path':str(evidence),'sha256':v.sha(evidence)}
    transition = d/'dispatcher.json'; v.write(transition, {'old_owner_absent':True,'new_owner_job_id':'123','retirement':ref})
    trans_ref = {'path':str(transition),'sha256':v.sha(transition)}
    handoff = d/'contact_map_handoffs/one.json'
    v.write(handoff, {'schema':'isg-contact-map-exclusive-handoff-v1','campaign_id':root.name,
        'unit_index':0,'source_plan':{'path':value['source_plan'],'sha256':v.sha(value['source_plan'])},
        'cluster_execution_excluded':True,'owner':'/root/templates_assets','mode':'never_submitted_retired','retirement':ref,'dispatcher_handoff':trans_ref,'evidence':[ref,trans_ref]})
    capacity = d/'capacity.json'
    v.write(capacity, {'schema':'isg-contact-map-cluster-first-v1','campaign_id':root.name,'observed_unix':time.time(),
        'accounts':{a:{'returncode':0,'pending_gpu_jobs':2,'evidence':ref} for a in ('minyeong','shiyuc')}})
    return handoff, capacity


def test_original_library_is_unchanged_and_queue_needs_owner_and_both_accounts(tmp_path):
    root, request, native, helper, lib = fixture(tmp_path); before = lib.read_bytes()
    job = v.prepare(root, request, 0, native, helper, 'one'); value = v.verify(job)
    assert lib.read_bytes() == before
    relocated = json.loads(Path(value['native_library']).read_text())
    relocated['object']['source'] = value['source_relocation']['before']
    assert relocated == json.loads(before)
    handoff, capacity = owner_and_capacity(root, job)
    jid = v.enqueue(job, root, handoff, capacity)
    assert v.enqueue(job, root, handoff, capacity) == jid
    assert not (root/'vast/config.json').exists()
    rec = json.loads((root/'vast/queue'/f'{jid}.json').read_text())
    assert rec['job_kind'] == 'native_dependency' and rec['cluster_first_observation']['sha256'] == v.sha(capacity)
    assert value['benchmark_credit'] == 0


@pytest.mark.parametrize('change', ['model', 'pose', 'height', 'arm', 'runtime'])
def test_changed_exact_inputs_are_rejected(tmp_path, change):
    root, job, lib = prepared(tmp_path); value = v.verify(job)
    if change == 'model':
        Path(value['native_library']).write_text('{}')
    else:
        p = Path(value['source_plan']); doc = json.loads(p.read_text())
        if change == 'pose': doc['units'][0]['pose_set']['digest'] = 'wrong'
        elif change == 'height': doc['units'][0]['height'] = .9
        elif change == 'arm': doc['units'][0]['arm'] = 'right'
        else: doc['units'][0]['native_runtime'] = 'wrong'
        v.write(p, doc)
    with pytest.raises(ValueError): v.verify(job)
    assert not (root/'vast/queue').exists()


def test_unowned_stale_or_single_account_overflow_is_rejected(tmp_path):
    root, job, lib = prepared(tmp_path); handoff, capacity = owner_and_capacity(root, job)
    original = json.loads(handoff.read_text()); bad = copy.deepcopy(original); bad['owner'] = '/root'; v.write(handoff,bad)
    with pytest.raises(ValueError,match='cluster owner'):v.enqueue(job,root,handoff,capacity)
    v.write(handoff,original); cap = json.loads(capacity.read_text()); cap['observed_unix'] -= 301; v.write(capacity,cap)
    with pytest.raises(ValueError,match='stale'):v.enqueue(job,root,handoff,capacity)
    cap['observed_unix'] = time.time(); del cap['accounts']['shiyuc']; v.write(capacity,cap)
    with pytest.raises(ValueError,match='both cluster'):v.enqueue(job,root,handoff,capacity)
    assert not (root/'vast/queue').exists()


def test_weights_and_wrong_native_kind_arm_are_rejected(tmp_path):
    root, request, native, helper, lib = fixture(tmp_path); doc = json.loads(request.read_text())
    unit = doc['units'][0]; unit['map_kind'] = 'push_blade'; v.write(request,doc)
    with pytest.raises(ValueError,match='kind/library/arm'):v.prepare(root,request,0,native,helper,'bad')
    unit['map_kind'] = 'axial_brace'; weights = lib.parent/'model.pt'; weights.write_text('weights')
    unit['source_files'][str(weights)] = v.sha(weights); v.write(request,doc)
    with pytest.raises(ValueError,match='weight payload'):v.prepare(root,request,0,native,helper,'bad')


@pytest.mark.parametrize('rc', [0, -11])
def test_process_status_is_not_positive_native_map_acceptance(tmp_path, monkeypatch, rc):
    root, job, lib = prepared(tmp_path); calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        if '--preflight' in argv:return SimpleNamespace(returncode=0,stdout='verified')
        kwargs['stdout'].write('GB_CONTACT_MAPS_DONE\n')
        if rc == -11:
            p = job/'result/maps/book/cmap__brace.json'; v.write(p, {'not_yet_verified':True}); p.with_suffix('.npz').write_bytes(b'opaque')
        return SimpleNamespace(returncode=rc)
    monkeypatch.setattr(v.subprocess,'run',run)
    assert v.run(job) == 2
    report = json.loads((job/'result/contact_map_result.json').read_text())
    assert report['returncode'] == rc and report['positive_native_reader_required']
    assert report['benchmark_credit'] == 0 and len(calls) == 2


def test_both_accounts_observed_but_only_one_has_pending_overflow(tmp_path):
    root, job, lib = prepared(tmp_path); handoff, capacity = owner_and_capacity(root, job)
    cap = json.loads(capacity.read_text()); cap['accounts']['shiyuc']['pending_gpu_jobs'] = 0; v.write(capacity,cap)
    assert v.validate_capacity(capacity,root)['sha256'] == v.sha(capacity)
    cap['accounts']['minyeong']['pending_gpu_jobs'] = 0; v.write(capacity,cap)
    with pytest.raises(ValueError,match='no pending GPU overflow'):v.validate_capacity(capacity,root)


def test_typed_contact_identity_dispatches_as_dependency_without_training_release(tmp_path, monkeypatch):
    from scripts.campaign_20261110 import production_gate as gate
    root, job, _ = prepared(tmp_path)
    assert 'original_entry' not in v.verify(job)
    decision = gate.check_job(root, job, kind='native_dependency')
    assert decision['allowed'] and decision['category'] == 'native_dependency'
    assert decision['class_ids'] == ['alignment-axial_02'] and decision['benchmark_credit'] == 0
    assert decision['configuration_ids'] == ['configuration_one']
    assert not gate.check_job(root, job, kind='collection')['allowed']
    monkeypatch.setattr(gate, 'PARKED_CLASSES', {'alignment-axial_02'})
    assert gate.check_job(root, job, kind='native_dependency')['category'] == 'owner_blocked'


@pytest.mark.parametrize('change', ['missing_consumer', 'changed_manifest', 'production_credit'])
def test_typed_dependency_gate_rejects_unverifiable_identity(tmp_path, change):
    from scripts.campaign_20261110 import production_gate as gate
    root, job, _ = prepared(tmp_path); value = v.verify(job)
    if change == 'changed_manifest':
        v.write(value['unit']['selected_manifest']['path'], {'campaign_id':root.name,'configurations':[]})
    elif change == 'missing_consumer':
        value['unit']['consumers'] = ['not_selected']; v.write(job/'native_inputs.json',value)
    else:
        value['benchmark_credit'] = 1; v.write(job/'native_inputs.json',value)
    assert not gate.check_job(root, job, kind='native_dependency')['allowed']


def copied_fixture(tmp_path):
    root, request, native, helper, lib = fixture(tmp_path)
    doc = json.loads(request.read_text()); unit = doc['units'][0]
    model = Path(unit['model_dir'])
    original_model = Path('/vision/u/shiyuc/private/campaign_20261110/glib/models')/model.name
    original_library = original_model.parent.parent.parent/'brace/models'/model.name/'library.json'
    library = json.loads(lib.read_text()); library['object']['source'] = str(original_model)
    v.write(lib, library); unit['object'] = copy.deepcopy(library['object'])
    unit['source_library']['sha256'] = v.sha(lib)
    unit['original_source_library'] = {'path':str(original_library), 'sha256':v.sha(lib)}
    files = {str(original_library):{'destination':str(lib),'sha256':v.sha(lib)}}
    for p in model.iterdir():
        files[str(original_model/p.name)] = {'destination':str(p),'sha256':v.sha(p)}
    proof = root/'training_preparation/readcopy/verified_copy.json'
    v.write(proof, {'schema':'isg-internal-exact-dependency-readcopy-v1','campaign_id':root.name,
                    'all_bytes_identical':True,'local_download':False,'files':files})
    unit['readcopy_proof'] = {'path':str(proof),'sha256':v.sha(proof)}
    unit['source_files'][str(proof)] = v.sha(proof)
    v.write(request,doc)
    return root, request, native, helper, lib, proof


def test_exact_internal_copy_preserves_original_source_and_library_bytes(tmp_path):
    root, request, native, helper, lib, proof = copied_fixture(tmp_path)
    before = lib.read_bytes()
    job = v.prepare(root,request,0,native,helper,'copied')
    value = v.verify(job)
    assert lib.read_bytes() == before
    assert value['source_model_identity']['mode'] == 'verified_internal_readcopy'
    assert value['source_relocation']['before'] == json.loads(before)['object']['source']
    assert not (root/'vast/queue').exists()


@pytest.mark.parametrize('change', ['missing_proof','wrong_entry','wrong_original_library','missing_geometry',
                                   'wrong_geometry_hash','wrong_destination','unverified_copy'])
def test_unbound_internal_copy_cannot_prepare_transport(tmp_path, change):
    root, request, native, helper, lib, proof = copied_fixture(tmp_path)
    doc = json.loads(request.read_text()); unit = doc['units'][0]; data=json.loads(proof.read_text())
    if change == 'missing_proof': unit.pop('readcopy_proof')
    elif change == 'wrong_entry': unit['entry_id']='another_entry'
    elif change == 'wrong_original_library': unit['original_source_library']['path'] += '.other'
    else:
        geometry=next(k for k in data['files'] if k.endswith('/geometry.obj'))
        if change == 'missing_geometry': del data['files'][geometry]
        elif change == 'wrong_geometry_hash': data['files'][geometry]['sha256']='0'*64
        elif change == 'wrong_destination': data['files'][unit['original_source_library']['path']]['destination']=unit['model_dir']+'/library.json'
        else: data['all_bytes_identical']=False
        v.write(proof,data);unit['readcopy_proof']['sha256']=v.sha(proof);unit['source_files'][str(proof)]=v.sha(proof)
    v.write(request,doc)
    with pytest.raises(ValueError): v.prepare(root,request,0,native,helper,'rejected')
    assert not (root/'out/vast_contact_map/rejected').exists()


def test_copy_proof_change_invalidates_prepared_payload(tmp_path):
    root, request, native, helper, lib, proof = copied_fixture(tmp_path)
    job=v.prepare(root,request,0,native,helper,'copied')
    data=json.loads(proof.read_text());data['files'].clear();v.write(proof,data)
    with pytest.raises(ValueError):v.verify(job)
