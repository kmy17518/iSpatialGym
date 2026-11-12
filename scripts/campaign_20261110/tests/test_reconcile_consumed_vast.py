import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.campaign_20261110 import reconcile_consumed_vast as m
from scripts.campaign_20261110 import vast_prebuild as v
from scripts.campaign_20261110 import vast_map_prebuild as maps
from scripts.campaign_20261110.tests.test_vast_prebuild import attachment_fixture
from scripts.campaign_20261110.tests.test_vast_map_prebuild import consumed_fixture


def setup(root, map_route=False):
    tickets, native, helper, _, ticket = (consumed_fixture if map_route else attachment_fixture)(root)
    ticket['expected_account'] = 'minyeong'
    v.write(tickets, {'schema': 'isg-preparation-dispatch-tickets-v1', 'jobs': [ticket]})
    job = (maps.prepare_consumed if map_route else v.prepare_consumed_attachment)(root, tickets, 0, native, helper, 'attachment')
    ref = lambda p: {'path': str(p), 'sha256': v.sha(p)}
    admission = root / 'training_preparation/admission.json'
    v.write(admission, {'schema': 'isg-consumed-preparation-admission-v1', 'approved_manifests': [
        {'manifest': ref(tickets), 'computation_kind': 'reach_map' if map_route else 'grasp_library', 'consumer_kind': 'tabletop' if map_route else 'attachment_free',
         'account': 'minyeong', 'claim_namespace': str(Path(ticket['claim_path']).parent),
         'receipt_namespace': str(Path(ticket['submission_path']).parent)}]})
    route = root / 'training_preparation/route.json'
    v.write(route, {'schema': 'isg-consumed-map-vast-route-v1' if map_route else 'isg-consumed-library-vast-route-v1', 'source_manifest': ref(tickets),
                   'jobs': [{'source_index': 0, 'physical_dedup_key': ticket['dedup_key'],
                             'claim_path': ticket['claim_path'], 'submission_path': ticket['submission_path'],
                             'job_dir': str(job), 'native_inputs': ref(job / 'native_inputs.json')}]})
    capacity = {'status': 'observed', 'observed_at': 1000, 'accounts_checked': ['minyeong', 'shiyuc'],
                'account_slots': {'minyeong': 2, 'shiyuc': 20}, 'compatible_available_slots': 4}
    status = root / 'infra/coordinator_dispatch/status_minyeong.json'; v.write(status, {'capacity': capacity})
    metrics = {'observed_at_unix': 1000, 'account_listing_complete': True, 'spend_estimate_usd': 1000,
               'future_compute_commitment_usd': 0, 'transfer_reserve_usd': 300,
               'instances': [{'id': 7, 'gpu_name': 'RTX 4090', 'num_gpus': 1, 'actual_status': 'running',
                              'worker_observation_current': True, 'worker_phase': 'ready', 'heartbeat_age_seconds': 3,
                              'assigned_jobs': [], 'staging_jobs': [], 'heartbeat_running_jobs': []}]}
    v.write(root / 'vast/metrics.json', metrics); v.write(root / 'vast/config.json', {'budget_usd': 3000, 'dispatch_job_ids': ['old-held']})
    policy = {'schema': 'isg-consumed-library-intake-v1', 'owner': '/root', 'budget_cap_usd': 3000,
              'production_authorized': False, 'critical_reserved_physical_slots': 4,
              'compatible_gpu_models': ['RTX 4090'], 'default_gpu': 'RTX 4090',
              'max_new_jobs_per_cycle': 32, 'max_examined_per_cycle': 256, 'max_cycle_seconds': 45,
              'admission_policy': ref(admission), 'sources': [{'enabled': True, 'route': ref(route)}],
              'cluster_status_path': str(status)}
    path = root / 'training_preparation/intake_policy.json'; v.write(path, policy)
    return path, ticket, policy, capacity, metrics


def allowed(*args, **kwargs): return {'allowed': True}


def test_consumed_map_overflow_uses_exact_claim_and_does_not_reactivate(tmp_path):
    path, ticket, policy, capacity, _ = setup(tmp_path, map_route=True)
    capacity.update(account_slots={'minyeong': 0, 'shiyuc': 48}, compatible_available_slots=48)
    v.write(policy['cluster_status_path'], {'capacity': capacity})
    result = m.cycle(tmp_path, path, v.sha(path), execute=True, timestamp=1000, gate=allowed)
    assert len(result['submitted']) == 1 and result['ready'][0]['gpu'] == 'RTX 4090'
    receipt = json.loads(Path(ticket['submission_path']).read_text())
    queue = json.loads((tmp_path / 'vast/queue' / (receipt['job_id'] + '.json')).read_text())
    assert queue['time_s'] == 1800 and queue['physical_dedup_key'] == ticket['dedup_key']
    assert queue['script'].endswith('job_map_prebuild.sh')
    v.write(tmp_path / 'vast/config.json', {'budget_usd': 3000, 'dispatch_job_ids': []})
    assert m.cycle(tmp_path, path, v.sha(path), execute=True, timestamp=1000, gate=allowed)['submitted'] == []
    assert json.loads((tmp_path / 'vast/config.json').read_text())['dispatch_job_ids'] == []


def test_consumed_map_claimed_by_cluster_is_skipped(tmp_path):
    path, ticket, _, _, _ = setup(tmp_path, map_route=True)
    claim = Path(ticket['claim_path']); claim.parent.mkdir(parents=True); claim.write_text('cluster assignment')
    result = m.cycle(tmp_path, path, v.sha(path), execute=True, timestamp=1000, gate=allowed)
    assert not result['submitted'] and result['results'][0]['status'] == 'existing_physical_claim_or_receipt'
    assert claim.read_text() == 'cluster assignment'


def test_dry_run_prefers_live_idle_worker_and_never_claims_or_enqueues(tmp_path):
    path, ticket, _, _, _ = setup(tmp_path)
    result = m.cycle(tmp_path, path, v.sha(path), timestamp=1000, gate=allowed)
    assert result['ready'][0]['gpu'] == 'RTX 4090' and not result['submitted']
    assert not Path(ticket['claim_path']).exists() and not (tmp_path / 'vast/queue').exists()
    assert json.loads((tmp_path / 'vast/config.json').read_text())['dispatch_job_ids'] == ['old-held']


def test_cluster_ownership_skipped_and_never_reactivated(tmp_path):
    path, ticket, _, _, _ = setup(tmp_path)
    claim = Path(ticket['claim_path']); claim.parent.mkdir(parents=True); claim.write_text('existing cluster allocation')
    result = m.cycle(tmp_path, path, v.sha(path), execute=True, timestamp=1000, gate=allowed)
    assert len(result['results']) == 1 and result['results'][0]['status'] == 'existing_physical_claim_or_receipt'
    assert claim.read_text() == 'existing cluster allocation' and not (tmp_path / 'vast/queue').exists()


def test_resumed_cycle_preserves_removed_activation_hold(tmp_path):
    path, _, _, _, _ = setup(tmp_path)
    first = m.cycle(tmp_path, path, v.sha(path), execute=True, timestamp=1000, gate=allowed)
    assert len(first['activated']) == 1
    v.write(tmp_path / 'vast/config.json', {'budget_usd': 3000, 'dispatch_job_ids': []})
    second = m.cycle(tmp_path, path, v.sha(path), execute=True, timestamp=1000, gate=allowed)
    assert second['submitted'] == second['activated'] == []
    assert json.loads((tmp_path / 'vast/config.json').read_text())['dispatch_job_ids'] == []


def test_backend_race_and_ambiguous_enqueue_preserve_claims(tmp_path):
    path, ticket, _, _, _ = setup(tmp_path)
    def race(*args, **kwargs):
        claim = Path(ticket['claim_path']); claim.parent.mkdir(parents=True); claim.write_text('cluster won')
        return {'allowed': True}
    result = m.cycle(tmp_path, path, v.sha(path), execute=True, timestamp=1000, gate=race)
    assert not result['submitted'] and result['results'][0]['status'] == 'another_backend_claimed'
    assert Path(ticket['claim_path']).read_text() == 'cluster won'


def test_failed_enqueue_is_not_automatically_retried(tmp_path):
    path, ticket, _, _, _ = setup(tmp_path)
    def failure(*args, **kwargs): raise OSError('queue unavailable after claim')
    fake = SimpleNamespace(verify=v.verify, enqueue=failure)
    first = m.cycle(tmp_path, path, v.sha(path), execute=True, timestamp=1000, gate=allowed, transport=fake)
    assert first['errors'] and Path(ticket['claim_path']).exists()
    second = m.cycle(tmp_path, path, v.sha(path), execute=True, timestamp=1000, gate=allowed)
    assert second['results'][0]['status'] == 'existing_physical_claim_or_receipt' and not second['submitted']


@pytest.mark.parametrize('change, reason', [
    ({'compatible_available_slots': 5}, 'cluster_headroom_available'),
    ({'observed_at': 700}, 'cluster_observation_unknown'),
    ({'account_slots': {'minyeong': 0}}, 'cluster_observation_unknown'),
    ({'status': 'unavailable'}, 'cluster_observation_unknown'),
])
def test_capacity_reserve_distinct_accounts_and_unknown(tmp_path, change, reason):
    _, _, policy, capacity, metrics = setup(tmp_path); capacity.update(change)
    result = m.routing_capacity(capacity, metrics, policy, 1000)
    assert not result['allow_overflow'] and result['reason'] == reason
    if reason == 'cluster_observation_unknown': assert result['usable_cluster_slots'] is None


def test_stopped_or_stale_worker_is_never_preferred_and_unknown_budget_blocks(tmp_path):
    _, _, policy, capacity, metrics = setup(tmp_path)
    metrics['instances'][0]['actual_status'] = 'exited'
    assert m.routing_capacity(capacity, metrics, policy, 1000)['compatible_idle_workers'] == []
    metrics['instances'][0]['actual_status'] = 'running'; metrics['instances'][0]['heartbeat_age_seconds'] = 301
    assert m.routing_capacity(capacity, metrics, policy, 1000)['compatible_idle_workers'] == []
    del metrics['future_compute_commitment_usd']
    assert m.routing_capacity(capacity, metrics, policy, 1000)['reason'] == 'campaign_budget_unknown'


def test_primary_only_ticket_overflows_when_secondary_alone_has_headroom(tmp_path):
    path, _, policy, capacity, metrics = setup(tmp_path)
    capacity.update(account_slots={'minyeong': 0, 'shiyuc': 48}, compatible_available_slots=48)
    v.write(Path(policy['cluster_status_path']), {'capacity': capacity})
    result = m.cycle(tmp_path, path, v.sha(path), timestamp=1000, gate=allowed)
    assert result['capacity']['usable_cluster_slots'] == 44
    assert len(result['ready']) == 1 and result['ready'][0]['gpu'] == 'RTX 4090'
    primary = m.routing_capacity(capacity, metrics, policy, 1000, eligible_accounts=['minyeong'])
    secondary = m.routing_capacity(capacity, metrics, policy, 1000, eligible_accounts=['shiyuc'])
    assert primary['allow_overflow'] and primary['usable_cluster_slots'] == 0
    assert not secondary['allow_overflow'] and secondary['usable_cluster_slots'] == 44


def test_physical_reserve_does_not_subtract_from_account_quota(tmp_path):
    _, _, policy, capacity, metrics = setup(tmp_path)
    capacity.update(account_slots={'minyeong': 2, 'shiyuc': 48}, compatible_available_slots=48)
    result = m.routing_capacity(capacity, metrics, policy, 1000, eligible_accounts=['minyeong'])
    assert result['usable_cluster_slots'] == 2 and not result['allow_overflow']


def test_unreviewed_or_changed_source_rejected_before_claim(tmp_path):
    path, ticket, policy, _, _ = setup(tmp_path)
    admission = Path(policy['admission_policy']['path']); value = json.loads(admission.read_text())
    value['approved_manifests'] = []; v.write(admission, value)
    policy['admission_policy']['sha256'] = v.sha(admission); v.write(path, policy)
    with pytest.raises(ValueError, match='outside reviewed admission'):
        m.cycle(tmp_path, path, v.sha(path), execute=True, timestamp=1000, gate=allowed)
    assert not Path(ticket['claim_path']).exists()
