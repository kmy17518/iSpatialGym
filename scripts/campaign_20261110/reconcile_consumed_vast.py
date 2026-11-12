"""Bounded root intake for reviewed consumed-library overflow.

One invocation is one resumable CPU cycle. Dry-run is the default. The existing
Vast controller alone assigns workers, rents capacity and enforces expenditure.
No scene-query adapter or previous activation list enters this route.
"""
import argparse
import fcntl
import json
import math
from pathlib import Path
import time

from scripts.campaign_20261110 import vast_prebuild as native
from scripts.campaign_20261110.infra_dispatch import shared_json
from scripts.campaign_20261110.production_gate import check_job
from scripts.campaign_20261110.storage import atomic_directory_lock


def checked(ref):
    native.require(native.sha(ref['path']) == ref['sha256'], 'Immutable input changed: ' + ref['path'])
    return json.loads(Path(ref['path']).read_text())


def routing_capacity(capacity, metrics, policy, timestamp):
    """Keep both account observations distinct and reserve recovery slots."""
    reserve = policy['critical_reserved_physical_slots']
    native.require(type(reserve) is int and reserve >= 0, 'Invalid critical reserve')
    fresh = (capacity.get('status') == 'observed' and
             0 <= timestamp - capacity.get('observed_at', 0) <= 180 and
             set(capacity.get('accounts_checked', [])) == {'minyeong', 'shiyuc'} and
             all(type(capacity.get('account_slots', {}).get(a)) is int for a in ('minyeong', 'shiyuc')) and
             type(capacity.get('compatible_available_slots')) is int)
    account_slots = {a: capacity.get('account_slots', {}).get(a) if fresh else None for a in ('minyeong', 'shiyuc')}
    usable = max(0, min(capacity['compatible_available_slots'], sum(account_slots.values())) - reserve) if fresh else None
    provider_fresh = metrics.get('account_listing_complete') and 0 <= timestamp - metrics.get('observed_at_unix', 0) <= 180
    idle = []
    if provider_fresh:
        for worker in metrics.get('instances', []):
            if (worker.get('actual_status') == 'running' and worker.get('worker_observation_current') is True and
                    worker.get('worker_phase') == 'ready' and not worker.get('assigned_jobs') and
                    not worker.get('staging_jobs') and not worker.get('heartbeat_running_jobs') and
                    type(worker.get('heartbeat_age_seconds')) in (int, float) and
                    0 <= worker['heartbeat_age_seconds'] + timestamp - metrics['observed_at_unix'] <= 300 and
                    worker.get('gpu_name') in policy['compatible_gpu_models']):
                idle.append({'id': str(worker['id']), 'gpu': worker['gpu_name'], 'slots': int(worker.get('num_gpus') or 1)})
    budget_keys = ('spend_estimate_usd', 'future_compute_commitment_usd', 'transfer_reserve_usd')
    budget_known = all(type(metrics.get(k)) in (int, float) and math.isfinite(metrics[k]) and metrics[k] >= 0 for k in budget_keys)
    exposure = sum(metrics[k] for k in budget_keys) if budget_known else None
    reason = ('cluster_observation_unknown' if not fresh else 'cluster_headroom_available' if usable > 0 else
              'provider_observation_unknown' if not provider_fresh else 'campaign_budget_unknown' if not budget_known else
              'campaign_budget_exhausted' if exposure >= 3000 else 'overflow_allowed')
    return {'reason': reason, 'allow_overflow': reason == 'overflow_allowed', 'usable_cluster_slots': usable,
            'account_slots': account_slots, 'critical_reserved_physical_slots': reserve,
            'compatible_idle_workers': idle, 'observed_exposure_usd': exposure,
            'rental_owner': 'existing sole Vast controller; no rentals from this intake'}


def reviewed_routes(root, policy):
    admission = checked(policy['admission_policy'])
    native.require(admission['schema'] == 'isg-consumed-preparation-admission-v1', 'Unsupported consumed scope')
    rows = []
    for source in policy['sources']:
        if not source.get('enabled'): continue
        route = checked(source['route'])
        native.require(route['schema'] == 'isg-consumed-library-vast-route-v1', 'Unsupported typed library route')
        approved = next((a for a in admission['approved_manifests'] if a['manifest'] == route['source_manifest']), None)
        native.require(approved is not None and approved['computation_kind'] == 'grasp_library' and
                       approved['consumer_kind'] == 'attachment_free', 'Library source is outside reviewed admission')
        original = checked(route['source_manifest'])
        for row in route['jobs']:
            ticket = original['jobs'][row['source_index']]
            key = row['physical_dedup_key']
            native.require(key == ticket['physical_dedup_key'] == ticket['dedup_key'], 'Physical library key differs')
            for field, namespace in (('claim_path', 'claim_namespace'), ('submission_path', 'receipt_namespace')):
                native.require(row[field] == ticket[field] and Path(row[field]).parent == Path(approved[namespace]) and
                               Path(row[field]).name == key + '.json', 'Shared physical claim differs')
            native.require(ticket['configuration_ids'] and ticket['expected_account'] == approved['account'], 'Selected consumer identity differs')
            rows.append({'route': source['route'], 'source_manifest': route['source_manifest'], 'ticket': ticket, **row})
    native.require(len({r['physical_dedup_key'] for r in rows}) == len(rows), 'Duplicate physical route')
    return rows


def cycle(root, policy_path, expected_sha, *, execute=False, timestamp=None, transport=native, gate=check_job):
    root = Path(root).resolve(); timestamp = time.time() if timestamp is None else timestamp
    policy_ref = {'path': str(policy_path), 'sha256': expected_sha}; policy = checked(policy_ref)
    native.require(policy['schema'] == 'isg-consumed-library-intake-v1' and policy['owner'] == '/root' and
                   policy['budget_cap_usd'] == 3000 and policy['production_authorized'] is False, 'Intake authority differs')
    native.require(1 <= policy['max_new_jobs_per_cycle'] <= 128 and 1 <= policy['max_examined_per_cycle'] <= 4096 and
                   1 <= policy['max_cycle_seconds'] <= 50, 'Unbounded CPU intake')
    rows = reviewed_routes(root, policy)
    status = json.loads(Path(policy['cluster_status_path']).read_text())
    metrics = json.loads((root / 'vast/metrics.json').read_text())
    capacity = routing_capacity(status.get('capacity', {}), metrics, policy, timestamp)
    output = root / 'infra/coordinator_dispatch/consumed_vast_intake'
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'cursor': 0, 'ever_activated': []}
    results = []; errors = []; submitted = []; ready = []; began = time.monotonic(); cursor = state['cursor']
    idle = [w['gpu'] for w in capacity['compatible_idle_workers'] for _ in range(w['slots'])]
    # This is a finite per-cycle buffer, never a fleet cap. The next cycle resumes.
    for offset in range(min(len(rows), policy['max_examined_per_cycle'])):
        if time.monotonic() - began >= policy['max_cycle_seconds']: break
        if len(ready) >= policy['max_new_jobs_per_cycle']: break
        row = rows[(cursor + offset) % len(rows)]; state['cursor'] = cursor + offset + 1
        claim = Path(row['claim_path']); receipt = Path(row['submission_path']); key = row['physical_dedup_key']
        if claim.exists() or receipt.exists():
            results.append({'physical_dedup_key': key, 'status': 'existing_physical_claim_or_receipt'}); continue
        if not capacity['allow_overflow']:
            results.append({'physical_dedup_key': key, 'status': capacity['reason']}); continue
        try:
            checked(row['native_inputs']); job = Path(row['job_dir']); value = transport.verify(job, root)
            native.require(value['kind'] == 'attachment' and value['scenes'] == [] and
                           value['physical_dedup_key'] == key and value['claim_path'] == str(claim) and
                           value['submission_path'] == str(receipt), 'Typed library route identity differs')
            decision = gate(root, job, kind='native_dependency'); native.require(decision['allowed'], str(decision))
            gpu = idle[len(ready)] if len(ready) < len(idle) else policy['default_gpu']
            ready.append({'physical_dedup_key': key, 'job_dir': str(job), 'gpu': gpu})
            if not execute: continue
            record = {'physical_dedup_key': key, 'backend': 'vast', 'dispatch_owner': '/root',
                      'owner': '/root/templates_assets', 'workstream': 'training preparation',
                      'stage': 'consumed attachment-free library', 'source': row['source_manifest'],
                      'job_dir': str(job), 'native_inputs_sha256': row['native_inputs']['sha256'],
                      'observed_at': timestamp, 'intake_policy': policy_ref, 'benchmark_credit': 0,
                      'production_released': False}
            claim.parent.mkdir(parents=True, exist_ok=True)
            try:
                with claim.open('x') as stream: json.dump({**record, 'status': 'submitting'}, stream)
            except FileExistsError:
                results.append({'physical_dedup_key': key, 'status': 'another_backend_claimed'}); continue
            shared_json(receipt, {**record, 'status': 'submitting'})
            jid = transport.enqueue(job, root, gpu=gpu)
            shared_json(receipt, {**record, 'status': 'submitted', 'job_id': str(jid)})
            submitted.append({'job_id': str(jid), 'receipt': str(receipt), 'physical_dedup_key': key})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({'physical_dedup_key': key, 'error': type(exc).__name__ + ': ' + str(exc),
                           'ambiguous_claims_preserved': True})
    activated = []
    if execute and submitted:
        with atomic_directory_lock(root / 'vast/config.dispatch.lockdir'):
            config_path = root / 'vast/config.json'; config = json.loads(config_path.read_text())
            native.require(config['budget_usd'] <= 3000, 'Campaign budget changed')
            for row in submitted:
                jid = row['job_id']
                # A removed previous activation is an owner hold, never a retry.
                if jid in state['ever_activated']: continue
                activated.append(jid)
            shared_json(output / 'activation_intent.json', {'observed_at': timestamp, 'jobs': submitted, 'policy': policy_ref})
            for field in ('dispatch_job_ids', 'dispatch_priority_job_ids'):
                config[field] = list(dict.fromkeys(list(config.get(field, [])) + activated))
            shared_json(config_path, config)
            state['ever_activated'] = sorted(set(state['ever_activated']) | set(activated))
    report = {'schema': 'isg-consumed-library-intake-status-v1', 'observed_at': timestamp, 'owner': '/root',
              'policy': policy_ref, 'execute': execute, 'capacity': capacity, 'total_routes': len(rows),
              'results': results, 'ready': ready, 'submitted': submitted, 'activated': activated, 'errors': errors,
              'benchmark_credit': 0, 'production_authorized': False}
    if execute: shared_json(state_path, state)
    shared_json(output / ('status.json' if execute else 'dry_run.json'), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'policy', 'policy-sha256'): parser.add_argument('--' + name, required=True)
    parser.add_argument('--execute-root-owned', action='store_true'); args = parser.parse_args()
    directory = Path(args.root) / 'infra/coordinator_dispatch/consumed_vast_intake'; directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'cycle.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return
        report = cycle(args.root, args.policy, args.policy_sha256, execute=args.execute_root_owned)
        print(json.dumps({'routes': report['total_routes'], 'ready': len(report['ready']), 'submitted': len(report['submitted']), 'errors': report['errors']}))


if __name__ == '__main__': main()
