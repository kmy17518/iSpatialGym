"""Read-only section-6 preparation projection for the existing campaign monitor.

Only derived reports are written. This module never claims, submits, retries,
changes quotas, acquires workers, or marks an artifact physically valid.
"""
import datetime as dt
import hashlib
import json
from pathlib import Path

TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL'}


def epoch(value):
    if isinstance(value, (float, int)):
        return float(value)
    try:
        return dt.datetime.fromisoformat(str(value).replace('Z', '+00:00')).timestamp()
    except (TypeError, ValueError):
        return None


def read(path, errors, *, optional=False, digest=None):
    path = Path(path)
    try:
        if optional and not path.exists():
            return {}
        raw = path.read_bytes()
        if digest and hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('immutable source changed')
        return json.loads(raw)
    except (OSError, ValueError) as exc:
        errors.append({'kind': 'preparation_observation_failed', 'source': str(path),
                       'owner': '/root', 'detail': str(exc)})
        return {}


def project(root, scheduled, now, previous=None):
    root = Path(root); directory = root/'infra/coordinator_dispatch'; errors = []
    previous = previous or {}
    policy = read(directory/'policy.json', errors)
    contract_ref = policy.get('preparation_scope', {}).get('policy_ref', {})
    contract = read(contract_ref['path'], errors, digest=contract_ref['sha256']) if contract_ref else {}
    approved = {r['manifest']['path']: r for r in contract.get('approved_manifests', [])}
    links = policy.get('preparation_scope', {}).get('normalized_manifests', [])
    links = {r['manifest']['path']: r for r in links}
    statuses = {}; status_rows = {}
    for account in ('minyeong', 'shiyuc'):
        status = read(directory/('status_'+account+'.json'), errors)
        statuses[account] = status
        observed = epoch(status.get('last_reconciliation_at'))
        if observed is None or now-observed > 60 or observed > now+5:
            errors.append({'kind': 'preparation_dispatcher_missed_two_cycles', 'account': account,
                           'last_reconciliation_at': observed, 'owner': '/root'})
        if 'requests' not in status or status.get('errors'):
            errors.append({'kind': 'preparation_ready_count_unknown', 'account': account,
                           'owner': '/root', 'detail': status.get('errors')})
        for row in status.get('requests', []):
            status_rows[row.get('claim_path')] = row
    by_id = {str(j['job_id']): j for j in scheduled}
    metrics = read(root/'vast/metrics.json', errors, optional=True)
    provider_fresh = 0 <= now-(epoch(metrics.get('observed_at_unix')) or 0) <= 180
    idle_vast = sum(int(w.get('num_gpus') or 1) for w in metrics.get('instances', [])
                    if provider_fresh and w.get('actual_status') == 'running'
                    and w.get('worker_observation_current') is True and w.get('worker_phase') == 'ready'
                    and not w.get('assigned_jobs') and not w.get('staging_jobs') and not w.get('heartbeat_running_jobs'))
    intake = read(directory/'consumed_vast_intake/status.json', errors, optional=True)
    vast_routable = {r['physical_dedup_key'] for r in intake.get('ready', [])}
    rows = []; seen = set(); complete = bool(contract_ref)
    for ref in policy.get('prepared_manifests', []):
        if ref['path'] not in links:
            continue
        source = approved.get(links[ref['path']]['source_manifest']['path'])
        if not source:
            complete = False
            errors.append({'kind': 'preparation_ready_count_unknown', 'owner': '/root', 'source': ref['path']})
            continue
        if not source['required_for_readiness']:
            # Optional historical builders are not preparation readiness gates
            # and must not generate a new verification campaign or stall alert.
            continue
        manifest = read(ref['path'], errors, digest=ref['sha256'])
        if 'tickets' not in manifest:
            complete = False
        for ticket in manifest.get('tickets', []):
            key = ticket['physical_dedup_key']
            if key in seen:
                continue
            seen.add(key)
            receipt = read(ticket['submission_path'], errors, optional=True)
            disposition = read(directory/'preparation_verification'/(key+'.json'), errors, optional=True)
            # Private inputs on the other account need not be exposed to this
            # observer. Their artifact-ready clock stays unknown; the reviewed
            # shared manifest and submission receipts still enumerate the queue.
            request = read(ticket['request']['path'], [], optional=True)
            jid = str(receipt.get('job_id') or '')
            backend = receipt.get('backend', 'cluster')
            job = by_id.get(jid, {})
            if backend == 'vast' and jid:
                state = read(root/'vast/state/jobs'/(jid+'.json'), errors, optional=True)
                status = state.get('status', 'UNKNOWN')
                terminal = status in {'synced', 'failed'}
                ended = epoch(state.get('synced_at') or state.get('ended_at') or state.get('finished_at'))
            else:
                status = job.get('execution_state', 'UNKNOWN')
                terminal = status in TERMINAL
                ended = epoch(job.get('ended_at'))
            old = next((r for r in previous.get('items', []) if r['physical_dedup_key'] == key), {})
            # First-observed terminal is a conservative fallback, never a heartbeat.
            terminal_at = epoch(disposition.get('terminal_at')) or ended or (old.get('terminal_at') if terminal else None)
            if terminal and terminal_at is None:
                terminal_at = now
            dispatched = epoch(receipt.get('submitted_at')) or (epoch(receipt.get('observed_at')) if jid else None)
            ready = not receipt and not Path(ticket['claim_path']).exists()
            account = ticket['expected_account']; dispatch = statuses[account]
            cap = dispatch.get('capacity', {}); cap_at = epoch(cap.get('observed_at'))
            slots = None
            if cap.get('status') == 'observed' and cap_at is not None and -5 <= now-cap_at <= 60:
                slots = min(cap.get('account_slots', {}).get(account, 0),
                            max(0, cap.get('compatible_available_slots', 0)-policy.get('critical_reserved_physical_slots', 4)))
            blocked = status_rows.get(ticket['claim_path'], {}).get('blocker')
            compatible = slots is not None and slots > 0 or idle_vast > 0 and key in vast_routable
            artifact_ready = epoch(request.get('artifact_ready_at') or request.get('observed_at'))
            row = {'physical_dedup_key': key, 'id': ticket['id'], 'owner': ticket['owner'],
                   'required_for_readiness': source['required_for_readiness'], 'consumer_kind': source['consumer_kind'],
                   'manifest': ref, 'account': account, 'backend': backend, 'job_id': jid or None,
                   'artifact_ready_at': artifact_ready, 'manifest_ready_at': epoch(ticket.get('ready_since')),
                   'submitted_at': dispatched, 'terminal_at': terminal_at,
                   'verification_started_at': epoch(disposition.get('verification_started_at')),
                   'verified_at': epoch(disposition.get('verified_at')),
                   'downstream_submitted_at': epoch(disposition.get('downstream_submitted_at')),
                   'execution_state': status, 'verification_status': disposition.get('status', 'unknown'),
                   'ready_unsubmitted': ready, 'compatible_available_slots': slots,
                   'compatible_capacity_available': bool(compatible), 'blocker': disposition.get('blocker') or blocked,
                   'next_check_at': disposition.get('next_check_at'), 'evidence': disposition.get('evidence', []),
                   'downstream_required': disposition.get('downstream_required'),
                   'downstream_ids': disposition.get('downstream_ids', [])}
            age = now-row['manifest_ready_at'] if row['manifest_ready_at'] is not None else None
            if ready and (age is None or slots is None):
                errors.append({'kind': 'preparation_ready_age_or_capacity_unknown', 'requirement': key, 'owner': '/root'})
            if ready and age is not None and age > 120 and compatible:
                errors.append({'kind': 'preparation_dispatch_stall', 'requirement': key, 'owner': '/root',
                               'waiting_seconds': age, 'ready_since': row['manifest_ready_at']})
            row['awaiting_verification'] = bool(terminal_at and not row['verified_at'] and disposition.get('status') != 'failed')
            documented = bool(disposition.get('owner') and (disposition.get('status') == 'running' and row['verification_started_at']
                              or disposition.get('blocker') and disposition.get('next_check_at')))
            if row['awaiting_verification'] and now-terminal_at > 300 and not documented:
                errors.append({'kind': 'preparation_terminal_unreviewed', 'requirement': key,
                               'owner': ticket['owner'], 'terminal_at': terminal_at})
            row['awaiting_downstream'] = bool(row['verified_at'] and row['downstream_required'] and not row['downstream_submitted_at'])
            if row['awaiting_downstream'] and now-row['verified_at'] > 120 and not row['blocker']:
                errors.append({'kind': 'preparation_downstream_stall', 'requirement': key,
                               'owner': ticket['owner'], 'verified_at': row['verified_at']})
            if receipt and not jid:
                errors.append({'kind': 'preparation_submission_requires_reconciliation', 'requirement': key,
                               'owner': '/root', 'receipt': ticket['submission_path']})
            rows.append(row)
    complete = complete and not any(e['kind'] in {'preparation_observation_failed', 'preparation_ready_count_unknown',
                                                 'preparation_dispatcher_missed_two_cycles'} for e in errors)
    required = [r for r in rows if r['required_for_readiness']]
    ready = [r for r in required if r['ready_unsubmitted']]
    recent = [r for r in required if r['submitted_at'] is not None and 0 <= now-r['submitted_at'] <= 600]
    waiting = [now-r['manifest_ready_at'] for r in ready if r['manifest_ready_at'] is not None]
    # An empty admitted queue cannot establish an empty campaign queue. Surface
    # an enumeration gap when capacity sits idle after the last real submission.
    any_capacity = idle_vast > 0 or any(
        s.get('capacity', {}).get('status') == 'observed'
        and 0 <= now-(epoch(s.get('capacity', {}).get('observed_at')) or 0) <= 60
        and s['capacity'].get('compatible_available_slots', 0) > policy.get('critical_reserved_physical_slots', 4)
        for s in statuses.values())
    inventory_complete = policy.get('preparation_inventory_complete') is True
    unexplained_idle = (any(r['compatible_capacity_available'] and not r['blocker'] for r in ready)
                        or not inventory_complete and any_capacity) and not recent
    idle_since = previous.get('unexplained_idle_since') or now if unexplained_idle else None
    if idle_since is not None and now-idle_since >= 300:
        errors.append({'kind': 'preparation_idle_capacity_stall', 'owner': '/root', 'idle_since': idle_since})
    if not inventory_complete:
        errors.append({'kind': 'preparation_campaign_ready_count_unknown', 'owner': '/root/templates_assets',
                       'detail': 'Admitted queue is enumerated separately; remaining selected consumers need incremental manifests or exact per-item blockers'})
    prior = {e['exception_key']: e for e in previous.get('exceptions', [])}
    for error in errors:
        key = '|'.join(str(error.get(k, '')) for k in ('kind', 'requirement', 'account', 'source'))
        error['exception_key'] = key
        error['first_raised_at'] = prior.get(key, {}).get('first_raised_at', now)
        error['root_response_due_at'] = error['first_raised_at']+300
    return {'schema': 'isg-preparation-flow-watchdog-v1', 'observed_at': now,
            'owner': 'existing read-only campaign monitor', 'root_response_owner': '/root',
            'admitted_queue_enumeration': 'complete' if complete else 'unknown',
            'campaign_ready_count': len(ready) if inventory_complete and complete else None,
            'campaign_inventory_enumeration': 'incomplete; independent ready batches continue',
            'optional_builders_excluded_from_required_flow': True,
            'ready_count': len(ready) if complete else None, 'known_ready_count': len(ready),
            'oldest_ready_age_seconds': max(waiting, default=None),
            'completed_awaiting_verification': sum(r['awaiting_verification'] for r in required),
            'completed_awaiting_downstream': sum(r['awaiting_downstream'] for r in required),
            'actual_submissions_last_ten_minutes': [{'job_id': r['job_id'], 'backend': r['backend'],
                                                   'submitted_at': r['submitted_at']} for r in recent],
            'idle_capacity_blockers': [{'requirement': r['physical_dedup_key'], 'blocker': r['blocker'],
                                       'next_check_at': r['next_check_at']} for r in ready if r['blocker']],
            'unexplained_idle_since': idle_since, 'exceptions': errors, 'items': rows}


def publish(root, scheduled, now):
    path = Path(root)/'infra/preparation_flow.json'
    old = read(path, [], optional=True)
    value = project(root, scheduled, now, old)
    temporary = path.with_suffix('.json.new'); temporary.write_text(json.dumps(value, indent=2)+'\n'); temporary.replace(path)
    return value
