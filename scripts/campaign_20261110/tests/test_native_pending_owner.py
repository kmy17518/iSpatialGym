import copy
import json
import time

import pytest

from scripts.campaign_20261110 import native_pending_owner as owner


@pytest.fixture
def registered(tmp_path, monkeypatch):
    root = tmp_path
    job = root/'out/vast_native_dependency/jar'
    job.mkdir(parents=True)
    (job/'native_inputs.json').write_text(json.dumps({'runtime_bounds': {
        'sampling_budget_s': 1200, 'subprocess_timeout_s': 1800, 'allocation_seconds': 2400},
        'argv': ['python', '--time-budget-s', '1200']}))
    path = root/'infra/capacity_recovery/native_pending_owner/jar/registration.json'
    path.parent.mkdir(parents=True)
    wrapper = root/'job.sh'
    wrapper.write_text('exact registered command')
    row = {'account': 'minyeong', 'job_id': '12345', 'wrapper': owner.reference(wrapper),
           'native_inputs': {'path': str(job/'native_inputs.json'), 'sha256': 'pinned'}}
    restriction = root/'secondary.json'
    restriction.write_text(json.dumps({'observed_at': time.time()}))
    row['secondary_transport_proof'] = owner.reference(restriction)
    path.write_text(json.dumps(row))
    monkeypatch.setattr(owner, 'verify_registration', lambda p, r: (row, job))
    monkeypatch.setattr(owner.native, 'verify', lambda *a: {})
    monkeypatch.setattr(owner.native, 'enqueue', lambda *a: '900000123')
    monkeypatch.setattr(owner.overflow_readiness, 'check', lambda *a: {'budget_usd': 3000})
    proof = {'job_id': '12345', 'scheduler': {'JobId': '12345', 'UserId': 'minyeong(100)',
             'Command': str(wrapper), 'JobState': 'PENDING', 'Restarts': '0',
             'Dependency': '(null)', 'Reason': 'Resources', 'TimeLimit': '00:40:00'},
             'accounting': {'state': 'PENDING', 'start': 'None', 'elapsed_seconds': 0,
                            'allocated_tres': ''}}
    return root, path, row, job, proof


def state_proof(proof, state, *, started=False):
    p = copy.deepcopy(proof)
    p['scheduler']['JobState'] = state
    p['accounting']['state'] = state
    if started:
        p['accounting'].update(start='2026-11-11T22:00:00', elapsed_seconds=1,
                               allocated_tres='cpu=8,gres/gpu=1')
    return p


def run(registered, proofs, **kwargs):
    root, path, _, _, _ = registered
    iterator = iter(proofs)
    return owner.recover(path, root, proof_fn=lambda j: next(iterator),
                         capacity_fn=lambda: {'accounts_checked': ['minyeong', 'shiyuc']}, **kwargs)


def test_pending_cancel_activate_exact_owner(registered):
    proof = registered[-1]
    cancelled, activated = [], []
    result = run(registered, [proof, proof, state_proof(proof, 'CANCELLED')],
                 cancel_fn=cancelled.append, activate_fn=lambda r, j, s: activated.append(j))
    assert cancelled == ['12345'] and activated == ['900000123']
    assert result['status'] == 'active'
    assert result['capacity']['accounts_checked'] == ['minyeong', 'shiyuc']


def test_budget_uses_full_typed_runtime_including_startup(registered):
    scheduler = owner.funded_scheduler(registered[3], {'TimeLimit': '00:30:00', 'Reason': 'Priority'})
    assert scheduler['TimeLimit'] == '00:40:00' and scheduler['Reason'] == 'Priority'


def test_pending_to_running_race_never_cancels(registered):
    proof = registered[-1]
    result = run(registered, [proof, state_proof(proof, 'RUNNING', started=True)],
                 cancel_fn=lambda j: pytest.fail('running cancellation'),
                 activate_fn=lambda *a: pytest.fail('duplicate allocation'))
    assert result['status'] == 'cluster_started_or_terminal'


@pytest.mark.parametrize('change', ['job', 'account', 'wrapper', 'dependency', 'array', 'restart'])
def test_wrong_identity_cannot_cancel(registered, change):
    p = copy.deepcopy(registered[-1])
    key, value = {'job': ('JobId', '54321'), 'account': ('UserId', 'shiyuc(200)'),
                  'wrapper': ('Command', '/other/job.sh'), 'dependency': ('Dependency', 'afterok:99'),
                  'array': ('ArrayJobId', '12345'), 'restart': ('Restarts', '1')}[change]
    p['scheduler'][key] = value
    with pytest.raises(ValueError):
        run(registered, [p], cancel_fn=lambda j: pytest.fail('wrong owner cancellation'))


def test_execution_evidence_blocks_cancel(registered):
    (registered[3]/'result').mkdir()
    with pytest.raises(ValueError, match='created output'):
        run(registered, [], cancel_fn=lambda j: pytest.fail('executed job cancellation'))


def test_cancellation_accounting_with_runtime_blocks_replacement(registered):
    proof = registered[-1]
    with pytest.raises(ValueError, match='zero-runtime'):
        run(registered, [proof, proof, state_proof(proof, 'CANCELLED', started=True)],
            cancel_fn=lambda j: None, activate_fn=lambda *a: pytest.fail('started replacement'))


def test_activation_failure_preserves_cancellation_and_recovers(registered):
    proof = registered[-1]
    def failed(*a):
        raise RuntimeError('temporary budget observation failure')
    with pytest.raises(RuntimeError):
        run(registered, [proof, proof, state_proof(proof, 'CANCELLED')],
            cancel_fn=lambda j: None, activate_fn=failed)
    saved = json.loads((registered[1].parent/'state.json').read_text())
    assert saved['status'] == 'cancelled_never_started'
    result = run(registered, [state_proof(proof, 'CANCELLED')],
                 cancel_fn=lambda j: pytest.fail('repeated cancellation'), activate_fn=lambda *a: {})
    assert result['status'] == 'active'


def test_crash_after_cancel_recovers_bound_intent(registered):
    root, path, row, job, proof = registered
    owner.native.write(path.parent/'state.json', {'status': 'registered', 'cancel_requested': True,
                                                  'before': proof, 'registration': owner.reference(path)})
    result = run(registered, [state_proof(proof, 'CANCELLED')],
                 cancel_fn=lambda j: pytest.fail('already cancelled'), activate_fn=lambda *a: {})
    assert result['status'] == 'active'


def test_swapped_saved_cancel_proof_rejected(registered):
    root, path, row, job, proof = registered
    wrong = state_proof(proof, 'CANCELLED')
    wrong['scheduler']['JobId'] = wrong['job_id'] = '99999'
    owner.native.write(path.parent/'state.json', {'status': 'cancelled_never_started',
                                                 'before': proof, 'after': wrong,
                                                 'registration': owner.reference(path)})
    with pytest.raises(ValueError, match='identity'):
        run(registered, [state_proof(proof, 'CANCELLED')], activate_fn=lambda *a: pytest.fail('wrong proof'))


def test_other_native_queue_owner_rejected(registered):
    (registered[3]/'vast_submission.json').write_text('{}')
    with pytest.raises(ValueError, match='different queue owner'):
        run(registered, [])


def test_crash_after_activation_recovers_running_output_without_duplicate(registered):
    root, path, row, job, proof = registered
    queue = root/'vast/queue/900000123.json'
    queue.parent.mkdir(parents=True)
    queue.write_text(json.dumps({'job_kind': 'native_dependency', 'job_dir': str(job),
                                 'native_inputs_sha256': row['native_inputs']['sha256']}))
    owner.native.write(job/'vast_submission.json', {'id': '900000123', 'record_sha256': owner.native.sha(queue)})
    owner.native.write(root/'vast/config.json', {'dispatch_job_ids': ['900000123']})
    owner.native.write(path.parent/'state.json', {'status': 'cancelled_never_started',
                    'before': proof, 'after': state_proof(proof, 'CANCELLED'),
                    'registration': owner.reference(path)})
    (job/'result').mkdir()
    result = run(registered, [state_proof(proof, 'CANCELLED')],
                 activate_fn=lambda *a: pytest.fail('duplicate activation'))
    assert result['recovered_after_activation'] is True


def test_changed_registration_rejected_after_cancel(registered):
    root, path, row, job, proof = registered
    owner.native.write(path.parent/'state.json', {'status': 'cancelled_never_started',
                    'before': proof, 'after': state_proof(proof, 'CANCELLED'),
                    'registration': {'path': str(path), 'sha256': 'another-source'}})
    with pytest.raises(ValueError, match='registration changed'):
        run(registered, [], activate_fn=lambda *a: pytest.fail('changed source'))


def test_stale_secondary_observation_blocks_cancellation(registered):
    root, path, row, job, proof = registered
    restriction = root/'stale_secondary.json'
    restriction.write_text(json.dumps({'observed_at': time.time()-301}))
    row['secondary_transport_proof'] = owner.reference(restriction)
    with pytest.raises(ValueError, match='refresh exact secondary'):
        run(registered, [proof], cancel_fn=lambda j: pytest.fail('stale private transport'))


@pytest.mark.parametrize('change', [None, 'wrapper_path', 'wrapper_hash', 'restriction_input',
                                    'restriction_path', 'restriction_age'])
def test_registration_binds_native_wrapper_and_exact_secondary_proof(tmp_path, monkeypatch, change):
    root = tmp_path
    wrapper = root/'job.sh'
    wrapper.write_text('native wrapper')
    dependency = root/'private.json'
    dependency.write_text('{}')
    job = root/'out/vast_native_dependency/jar'
    job.mkdir(parents=True)
    manifest = {'job_dir': str(job), 'original_wrapper': str(wrapper),
                'files': {str(wrapper): owner.native.sha(wrapper), str(dependency): owner.native.sha(dependency)}}
    if change == 'wrapper_path':
        manifest['original_wrapper'] = str(root/'other.sh')
    if change == 'wrapper_hash':
        manifest['files'][str(wrapper)] = 'wrong'
    owner.native.write(job/'native_inputs.json', manifest)
    native_ref = owner.reference(job/'native_inputs.json')
    restriction = {'account': 'shiyuc', 'compatible': False, 'failed_paths': [str(dependency)],
                   'native_inputs': native_ref, 'observed_at': time.time()}
    if change == 'restriction_input':
        restriction['native_inputs'] = {'path': '/other', 'sha256': 'other'}
    if change == 'restriction_path':
        restriction['failed_paths'] = ['/unrelated/private.json']
    if change == 'restriction_age':
        restriction['observed_at'] -= 301
    owner.native.write(root/'secondary.json', restriction)
    command = ['sbatch', '--gres=gpu:titanrtx:1', '--exclude=svl9', '-c', '8',
               '--mem', '48G', '-p', 'svl,viscam', str(wrapper)]
    owner.native.write(root/'submission.json', {'job_id': '12345', 'returncode': 0, 'command': command})
    row = {'schema': 'isg-native-pending-owner-v1', 'account': 'minyeong', 'job_id': '12345',
           'cluster_routes': ['minyeong_svl', 'minyeong_viscam'], 'secondary_transport_blocker': 'private input',
           'secondary_transport_proof': owner.reference(root/'secondary.json'), 'registered_at': time.time(),
           'submission': owner.reference(root/'submission.json'), 'wrapper': owner.reference(wrapper),
           'native_inputs': native_ref}
    owner.native.write(root/'registration.json', row)
    monkeypatch.setattr(owner.native, 'verify', lambda *a: manifest)
    if change:
        with pytest.raises(ValueError):
            owner.verify_registration(root/'registration.json', root)
    else:
        assert owner.verify_registration(root/'registration.json', root) == (row, job)
