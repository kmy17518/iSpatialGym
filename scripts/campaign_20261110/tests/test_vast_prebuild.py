import json
from pathlib import Path

import pytest

from scripts.campaign_20261110 import vast_prebuild as v


def fixture(root):
    native=root/'code/native'; helper=root/'code/helper'
    for p in [native/'.snapshot_commit',native/'scripts/grasp_bands/job_glib.sh',
              helper/'scripts/campaign_20261110/vast_prebuild.py',helper/'scripts/campaign_20261110/job_prebuild.sh']:
        p.parent.mkdir(parents=True,exist_ok=True); p.write_text('exact runtime')
    asset=native/'Behavior-1K/datasets/objects/model/usd/model.usd'; asset.parent.mkdir(parents=True); asset.write_text('asset')
    proof=root/'training_preparation/default.json'; v.write(proof,{'measured':True})
    row={'id':'model__s0.9','ready':True,'kind':'tabletop','constraint':'tabletop','scale':[.9]*3,
         'campaign_classes':['disembedding_07'],'asset_path':str(asset),'asset_sha256':v.sha(asset),
         'native_default_evidence':{'path':str(proof),'sha256':v.sha(proof)}}
    manifest=root/'training_preparation/jobs.json'; v.write(manifest,[row])
    request=root/'training_preparation/request.json'
    v.write(request,{'batches':[{'kind':'tabletop','jobs':str(manifest),'jobs_sha256':v.sha(manifest),'stages':'glib'}]})
    return request,native,helper,asset,manifest


def test_exact_recipe_copies_row_and_idempotently_enqueues_without_activation(tmp_path):
    request,native,helper,asset,manifest=fixture(tmp_path)
    before=manifest.read_bytes()
    job=v.prepare(tmp_path,request,'tabletop',0,native,helper,'one')
    value=v.verify(job,tmp_path)
    copied=json.loads(Path(value['selected_manifest']).read_text())
    copied[0]['native_default_evidence']['path']=value['proof_relocation']['original']
    assert copied==json.loads(before)
    assert manifest.read_bytes()==before and value['new_native_instances']==0
    jid=v.enqueue(job,tmp_path); assert v.enqueue(job,tmp_path)==jid
    assert not (tmp_path/'vast/config.json').exists()
    with pytest.raises(ValueError,match='already has evidence'):v.prepare(tmp_path,request,'tabletop',0,native,helper,'one')


def test_changed_native_asset_fails_before_execution(tmp_path):
    request,native,helper,asset,manifest=fixture(tmp_path)
    job=v.prepare(tmp_path,request,'tabletop',0,native,helper,'one')
    asset.write_text('other model')
    with pytest.raises(ValueError,match='asset changed'):v.verify(job,tmp_path)
    assert not (job/'result').exists()


def test_unproved_floating_and_block_design_do_not_allocate(tmp_path):
    request,native,helper,asset,manifest=fixture(tmp_path)
    with pytest.raises(ValueError,match='GraspGen'):v.prepare(tmp_path,request,'floating',0,native,helper,'one')
    rows=json.loads(manifest.read_text());rows[0]['campaign_classes']=['configuration_matching-block_design_01'];v.write(manifest,rows)
    req=json.loads(request.read_text());req['batches'][0]['jobs_sha256']=v.sha(manifest);v.write(request,req)
    with pytest.raises(ValueError,match='selection remains'):v.prepare(tmp_path,request,'tabletop',0,native,helper,'one')
    assert not (tmp_path/'out/vast_prebuild/one').exists()


def test_native_zero_exit_without_validation_output_is_failure(tmp_path,monkeypatch):
    request,native,helper,asset,manifest=fixture(tmp_path)
    job=v.prepare(tmp_path,request,'tabletop',0,native,helper,'one')
    class Result:returncode=0
    monkeypatch.setattr(v.subprocess,'run',lambda *a,**k:Result())
    assert v.run(job)==2
    report=json.loads((job/'result/prebuild_result.json').read_text())
    assert not report['native_validation_done'] and report['positive_native_reader_required']
    assert report['new_native_instances']==0


def attachment_fixture(root):
    _, native, helper, _, _ = fixture(root)
    asset = native / 'Behavior-1K/datasets/test/objects/plug/abc/usd/abc.encrypted.usd'
    asset.parent.mkdir(parents=True); asset.write_text('exact scaled model source')
    directory = root / 'training_preparation/attachment/exact'; directory.mkdir(parents=True)
    entry = {'id': 'plug__abc__s0.9__attachment_free', 'category': 'plug', 'model': 'abc', 'dataset': 'test',
             'scale': [.9] * 3, 'kind': 'attachment', 'constraint': 'attachment_free', 'rest_pose': 'canonical',
             'rest_hint_wxyz': None, 'legacy_key': None, 'asset_sha256': v.sha(asset)}
    manifest = directory / 'jobs.json'; v.write(manifest, [entry])
    physical = {'kind': 'grasp_library', 'job': entry, 'generator_files': [], 'arm': 'right'}
    key = v.hashlib.sha256(json.dumps(physical, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    ref = lambda path: {'path': str(path), 'sha256': v.sha(path)}
    inputs = [ref(asset), ref(manifest), ref(native / '.snapshot_commit')]
    prepared = directory / 'prepared.json'
    v.write(prepared, {'physical_dedup_key': key, 'physical_identity': physical, 'native_runtime': str(native),
                      'entry_id': entry['id'], 'configuration_ids': ['attachment_01:model_a'],
                      'native_argv': ['python', '-u', '-m', 'ispatialgym.grasping.pipeline', 'all', '--jobs', str(manifest),
                                      '--root', str(directory / 'glib'), '--shard', '0/1', '--workers', '8']})
    ownership = root / 'infra/coordination/preparation_dispatch/consumed_asset_builds_r1'
    ticket = {'computation_kind': 'grasp_library', 'consumer_kind': 'attachment_free', 'physical_dedup_key': key,
              'dedup_key': key, 'claim_path': str(ownership / 'claims' / (key + '.json')),
              'submission_path': str(ownership / 'receipts' / (key + '.json')), 'configuration_ids': ['attachment_01:model_a'],
              'input_refs': inputs, 'scripts': [], 'prepared': ref(prepared), 'runtime_ref': ref(native / '.snapshot_commit'),
              'native_time_seconds': 3300, 'hard_time_seconds': 3600, 'benchmark_credit': 0, 'production_authorized': False}
    tickets = directory.parent / 'tickets.json'
    v.write(tickets, {'schema': 'isg-preparation-dispatch-tickets-v1', 'jobs': [ticket]})
    return tickets, native, helper, manifest, ticket


def test_attachment_transports_exact_recipe_and_shared_claim_without_allocation(tmp_path):
    tickets, native, helper, manifest, ticket = attachment_fixture(tmp_path)
    job = v.prepare_consumed_attachment(tmp_path, tickets, 0, native, helper, 'attachment')
    value = v.verify(job)
    assert json.loads(Path(value['selected_manifest']).read_text()) == json.loads(manifest.read_text())
    assert value['claim_path'] == ticket['claim_path'] and value['scenes'] == []
    assert not Path(value['claim_path']).exists() and not (tmp_path / 'vast').exists()
    assert value['time_limit_s'] == 3300 and value['allocation_time_s'] == 3600
    jid = v.enqueue(job, tmp_path)
    record = json.loads((tmp_path / 'vast/queue' / (jid + '.json')).read_text())
    assert record['physical_dedup_key'] == ticket['dedup_key'] and record['time_s'] == 3600
    assert not Path(value['claim_path']).exists()


@pytest.mark.parametrize('change, message', [
    ({'claim_path': 'wrong'}, 'claim ownership'),
    ({'native_time_seconds': 9000}, 'bounds differ'),
    ({'consumer_kind': 'floating'}, 'unsupported consumed'),
    ({'dedup_key': '0' * 64}, 'physical library key'),
])
def test_attachment_rejects_incompatible_ticket_before_output(tmp_path, change, message):
    tickets, native, helper, _, ticket = attachment_fixture(tmp_path)
    ticket.update(change); v.write(tickets, {'schema': 'isg-preparation-dispatch-tickets-v1', 'jobs': [ticket]})
    with pytest.raises(ValueError, match=message):
        v.prepare_consumed_attachment(tmp_path, tickets, 0, native, helper, 'attachment')
    assert not (tmp_path / 'out/vast_prebuild/attachment').exists()


def test_attachment_changed_native_recipe_rejected_before_execution(tmp_path):
    tickets, native, helper, manifest, _ = attachment_fixture(tmp_path)
    job = v.prepare_consumed_attachment(tmp_path, tickets, 0, native, helper, 'attachment')
    rows = json.loads(manifest.read_text()); rows[0]['scale'] = [1.1] * 3; v.write(manifest, rows)
    with pytest.raises(ValueError, match='prebuild input changed'):
        v.verify(job)
    assert not (job / 'result').exists()
