from pathlib import Path
import json
import pytest

from scripts.campaign_20261110.infra_vast import cluster_return as cr


def fixture(tmp_path):
    root = tmp_path / 'campaign_20261110'
    jobdir = root / 'out/vast_map_prebuild/unit'
    plan = root / 'training_preparation/example_maps_dispatch_r1/plan.json'
    target = plan.parent / 'build_dispatch_r1'
    script = target / 'units/0/job.sh'
    cr.write(plan, {'runtime': 'frozen', 'units': []})
    script.parent.mkdir(parents=True)
    script.write_text('unchanged native map command\n')
    (target / 'pinned.sh').write_text('pinned run.py\n')
    (target / 'run.py').write_text('exact native map runner\n')
    library = jobdir / 'inputs/library.json'
    cr.write(library, {'positive': 6})
    unit = {'index': 0, 'script': str(script), 'script_sha256': cr.digest(script), 'output': str(script.parent)}
    for name, field in (('jobs.json', 'jobs_sha256'), ('anchors.json', 'anchors_sha256')):
        cr.write(script.parent/name, {})
        unit[field] = cr.digest(script.parent/name)
    prepared = target / 'prepared.json'
    cr.write(prepared, {'units': [unit], 'copied_library_files': {}})
    manifest = jobdir / 'native_inputs.json'
    cr.write(manifest, {'schema': 'isg-vast-map-prebuild-v1', 'new_native_instances': 0,
                       'production_released': False, 'unit_index': 0, 'source_plan': str(plan),
                       'files': {str(plan): cr.digest(plan)}, 'library_files': {str(library): cr.digest(library)}})
    job = {'id': '9001', 'status': 'queued', 'tries': 0, 'history': [], 'job_kind': 'native_dependency',
           'job_dir': str(jobdir), 'script': '/frozen/job_map_prebuild.sh', 'native_inputs_sha256': cr.digest(manifest)}
    queue = root / 'vast/queue/9001.json'
    cr.write(queue, job)
    capacity = root / 'vast/cluster_return/capacity/one.json'
    cr.write(capacity, {'observed_at': 1000, 'compatible_available_slots': 1, 'selected_job_ids': ['9001'], 'selected_accounts': {'9001': 'minyeong'}})
    ref = lambda p: {'path': str(p), 'sha256': cr.digest(p)}
    request = {'schema': cr.SCHEMA, 'campaign_id': root.name, 'job_id': '9001', 'account': 'minyeong',
               'observed_at': 1000, 'expires_at': 1120, 'queue': ref(queue), 'native_inputs': ref(manifest),
               'source_plan': ref(plan), 'prepared': ref(prepared), 'unit_index': 0, 'unit': unit, 'capacity': ref(capacity),
               'wrapper': ref(target / 'pinned.sh'), 'runner': ref(target / 'run.py')}
    request_path = root / 'vast/cluster_return/requests/9001_one.json'
    cr.write(request_path, request)
    return root, request, job, library, request_path


@pytest.mark.parametrize('owner', ['stage', 'heartbeat', 'remote'])
def test_existing_owner_prevents_cluster_return(tmp_path, owner):
    root, request, job, _, _ = fixture(tmp_path)
    owners = [set(), set(), set()]
    owners[['stage', 'heartbeat', 'remote'].index(owner)].add('9001')
    with pytest.raises(ValueError, match='owner'):
        cr.validate(root, request, job, *owners, 1001)
    assert job['status'] == 'queued'


@pytest.mark.parametrize('change', [{'tries': 1}, {'instance': 'worker'}, {'status': 'assigned'}, {'history': ['attempt']}])
def test_any_previous_assignment_prevents_return(tmp_path, change):
    root, request, job, _, _ = fixture(tmp_path)
    job.update(change)
    with pytest.raises(ValueError, match='never-assigned'):
        cr.validate(root, request, job, set(), set(), set(), 1001)


def test_changed_native_bytes_are_rejected_without_claim(tmp_path):
    root, request, job, library, _ = fixture(tmp_path)
    library.write_text('changed payload')
    with pytest.raises(ValueError, match='payload changed'):
        cr.validate(root, request, job, set(), set(), set(), 1001)
    assert job['status'] == 'queued'


def test_expired_capacity_and_existing_cluster_output_are_rejected(tmp_path):
    root, request, job, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='expired'):
        cr.validate(root, request, job, set(), set(), set(), 1200)
    Path(request['unit']['output'], 'started.json').write_text('{}')
    with pytest.raises(ValueError, match='already started'):
        cr.validate(root, request, job, set(), set(), set(), 1001)


def test_acknowledgement_follows_durable_claim_and_recovers_once(tmp_path, monkeypatch):
    root, request, job, _, request_path = fixture(tmp_path)
    saved = []
    original_write = cr.write
    def fail_first_ack(path, value):
        if Path(path).parent.name == 'acknowledgements':
            assert saved and saved[-1]['status'] == 'cluster_reserved'
            raise OSError('simulated interrupted acknowledgement')
        original_write(path, value)
    monkeypatch.setattr(cr, 'write', fail_first_ack)
    report = cr.reconcile(root, {'9001': job}, {}, {}, set(), lambda j: saved.append(json.loads(json.dumps(j))), now=1001)
    assert report['deferred'] and not report['rejected'] and len(saved) == 1
    assert not (root / 'vast/cluster_return/rejected' / request_path.name).exists()
    monkeypatch.setattr(cr, 'write', original_write)
    report = cr.reconcile(root, {'9001': job}, {}, {}, set(), lambda j: saved.append(j), now=1300)
    assert report['acknowledged'] == ['9001'] and len(saved) == 1
    assert not cr.has_pending(root)
    repeated = cr.reconcile(root, {'9001': job}, {}, {}, set(), lambda j: saved.append(j), now=1301)
    assert repeated['acknowledged'] == [] and len(saved) == 1


def test_capacity_is_shared_across_accounts_and_snapshots(tmp_path):
    root, request, job, _, _ = fixture(tmp_path)
    other = {'job_id': '9002', 'account': 'shiyuc', 'claimed_at': 999,
             'capacity': {'path': 'a-different-snapshot'}}
    with pytest.raises(ValueError, match='aggregate'):
        cr.validate(root, request, job, set(), set(), set(), 1001, claims=[other])
    # The prior allocation is already represented in the fresh scheduler capacity.
    output = tmp_path / 'other_output'
    other.update(unit={'output': str(output)}, unit_index=2)
    cr.write(output / 'started.json', {'started_at': 999, 'index': 2, 'job_id': '12345'})
    assert cr.validate(root, request, job, set(), set(), set(), 1001, claims=[other])['job_id'] == '9001'


def test_same_job_cannot_be_selected_for_both_accounts(tmp_path):
    root, request, job, _, _ = fixture(tmp_path)
    p = Path(request['capacity']['path'])
    data = json.loads(p.read_text()); data['selected_accounts']['9001'] = 'shiyuc'
    cr.write(p, data); request['capacity']['sha256'] = cr.digest(p)
    with pytest.raises(ValueError, match='unique account'):
        cr.validate(root, request, job, set(), set(), set(), 1001)


def test_submission_requires_ack_and_preserves_uncertain_intent(tmp_path, monkeypatch):
    from scripts.campaign_20261110.infra_vast import cluster_return_client as client
    root, request, job, _, path = fixture(tmp_path)
    # No acknowledgement: no scheduler invocation, even though a request exists.
    calls = []
    monkeypatch.setattr(client.subprocess, 'run', lambda *a, **k: calls.append(a))
    assert client.submit(root, 'minyeong')['submitted'] == []
    assert calls == []
    cr.reconcile(root, {'9001': job}, {}, {}, set(), lambda j: cr.write(root/'vast/state/jobs/9001.json', j), now=1001)
    attempt = root/'infra/capacity_recovery/cluster_return_client_r1'/(path.stem+'_attempt.json')
    cr.write(attempt, {'status': 'submitting', 'uncertain_scheduler_result': True})
    result = client.submit(root, 'minyeong')
    assert result['preserved_submission_intents'] == [str(attempt)]
    assert calls == []


def test_changed_launch_code_prevents_claim(tmp_path):
    root, request, job, _, _ = fixture(tmp_path)
    Path(request['runner']['path']).write_text('changed run.py')
    with pytest.raises(ValueError, match='launch code changed'):
        cr.validate(root, request, job, set(), set(), set(), 1001)


def test_even_unit_can_use_available_secondary_immutable_closure(tmp_path, monkeypatch):
    from scripts.campaign_20261110.infra_vast import cluster_return_client as client
    root, request, job, _, _ = fixture(tmp_path)
    source = Path(request['prepared']['path']).parent
    secondary = tmp_path / 'secondary/build_dispatch_return_r1'
    secondary.mkdir(parents=True)
    script = secondary / 'job.sh'; script.write_text('exact same native map command')
    unit = {'index': 0, 'script': str(script), 'script_sha256': cr.digest(script),
            'output': str(secondary / 'units/0')}
    cr.write(secondary / 'prepared.json', {'units': [unit]})
    monkeypatch.setattr(cr, 'prepared_targets', lambda r, account, p:
                        (source,) if account == 'minyeong' else (secondary,))
    inputs = {'unit_index': 0, 'source_plan': request['source_plan']['path']}
    assert client.owned_preparation(root, inputs, {'minyeong': 0, 'shiyuc': 1}) == ('shiyuc', secondary, unit)
    assert client.owned_preparation(root, inputs, {'minyeong': 1, 'shiyuc': 1})[0] == 'minyeong'
    assert client.owned_preparation(root, inputs, {'minyeong': 0, 'shiyuc': 0}) is None
    cr.write(Path(unit['output']) / 'started.json', {'job_id': 123})
    with pytest.raises(ValueError, match='already executed'):
        client.owned_preparation(root, inputs, {'minyeong': 0, 'shiyuc': 1})


def test_separate_return_manifest_requires_exact_owned_location_and_hash(tmp_path):
    import shutil
    root, request, job, _, _ = fixture(tmp_path)
    original = Path(request['prepared']['path']).parent
    original_digest = cr.digest(original / 'prepared.json')
    target = original.with_name('build_dispatch_return_r1')
    shutil.copytree(original, target)
    unit = dict(request['unit'])
    unit['output'] = str(target / 'units/0'); unit['script'] = str(target / 'units/0/job.sh')
    cr.write(target / 'prepared.json', {'units': [unit], 'copied_library_files': {}})
    request['unit'] = unit
    for field, name in [('prepared', 'prepared.json'), ('wrapper', 'pinned.sh'), ('runner', 'run.py')]:
        request[field] = {'path': str(target/name), 'sha256': cr.digest(target/name)}
    assert cr.validate(root, request, job, set(), set(), set(), 1001)['prepared'] == request['prepared']
    assert cr.digest(original / 'prepared.json') == original_digest
    moved = target.with_name('unreviewed_other_directory')
    shutil.copytree(target, moved)
    request['prepared'] = {'path': str(moved/'prepared.json'), 'sha256': cr.digest(moved/'prepared.json')}
    with pytest.raises(ValueError, match='preparation changed'):
        cr.validate(root, request, job, set(), set(), set(), 1001)
