"""Reconcile unused production tickets from their actual producer records.

This worker never dispatches, retries, cancels or accepts instances. It observes
only explicitly registered tickets. Prepared or incomplete producer work cannot
be finalized, and missing evidence becomes an owned action rather than credit.
"""
from copy import deepcopy
from datetime import datetime
import argparse
import json
from pathlib import Path
import time

from .contracts import digest, file_digest, require, write_json_atomic
from .storage import atomic_directory_lock
from . import training_diversity as tickets


def reference(path):
    return {'path': str(path), 'sha256': file_digest(path)}


def register(root, ticket_ref, producer_state):
    """The caller retains producer ownership; register its stable state path."""
    root = Path(root).resolve(); state = tickets._read_state(root)
    ticket = tickets._read_reference(root, ticket_ref); request = ticket['request_id']
    binding = state.get('controllers', {}).get(request)
    require(binding and binding['ticket'] == ticket_ref, 'ticket has no registered producer')
    source = tickets._owned(root, producer_state)
    require(source.is_relative_to(root/'controllers'), 'producer state outside controllers')
    value = {'schema': 'isg-training-ticket-observer-registration-v1', 'campaign_id': tickets.CAMPAIGN,
             'ticket': ticket_ref, 'producer_state': str(source), 'controller': binding,
             'dispatch_authorization': False}
    path = root/'controllers/training_ticket_observer/registrations'/(request+'.json')
    if path.exists(): require(json.loads(path.read_text()) == value, 'ticket observer registration changed')
    else: write_json_atomic(path, value)
    return reference(path)


def _fresh(value, now):
    if isinstance(value, str): value = datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
    require(type(value) in (int, float) and -30 <= now-value <= 120, 'producer observation stale or missing')


def _records(state):
    rows = list(state.get('records', {}).values()) + list(state.get('queues', []))
    for seed in state.get('seeds', {}).values(): rows.extend(seed.get('queues', []))
    require(all(isinstance(row, dict) for row in rows), 'malformed producer record inventory')
    yield from rows


def _copy_json(root, source, destination):
    source = tickets._owned(root, source); before = reference(source)
    data = json.loads(source.read_text())
    require(reference(source) == before, 'source changed during observation')
    write_json_atomic(destination, data)
    return data, reference(destination), before


def derive(root, registration, directory, *, now=None):
    """Build a candidate closure; actual scheduler checks still precede commit."""
    from .controller_saved_allocations import adopt_receipt
    from .controller_held import queue_closures
    from .goal_groups import read_env
    from .acceptance import audit_job, report_passed
    root = Path(root).resolve(); now = time.time() if now is None else now
    require(registration.get('schema') == 'isg-training-ticket-observer-registration-v1'
            and registration.get('campaign_id') == tickets.CAMPAIGN, 'foreign observer registration')
    directory = tickets._owned(root, directory); directory.mkdir(parents=True, exist_ok=False)
    ticket_ref = registration['ticket']; ticket = tickets._read_reference(root, ticket_ref)
    state = tickets._read_state(root); request = ticket['request_id']
    require(state.get('controllers', {}).get(request) == registration['controller'], 'producer ownership changed')
    cp = Path(registration['controller']['path']); controller, _, before = _copy_json(root, cp, directory/'controller_before.json')
    tickets._controller_scope(controller, ticket_ref, ticket)
    queue = tickets._read_reference(root, controller['queue'])
    require(queue['config_sha256'] == ticket['config_sha256'] and
            file_digest(tickets._owned(root, queue['config_path'])) == ticket['config_sha256'], 'ticket queue config differs')
    require(digest(controller['command']) == controller['command_sha256'], 'producer command changed')
    producer, producer_ref, producer_original = _copy_json(root, registration['producer_state'], directory/'producer.json')
    _fresh(producer.get('updated_at'), now)
    matches = [r for r in _records(producer) if Path(r.get('queue', '/')).resolve() == Path(controller['queue']['path']).resolve()]
    require(len(matches) == 1, 'exact producer queue missing or duplicated')
    record = matches[0]
    require(record.get('status') == 'complete', 'producer dispatch/acceptance reconciliation unfinished')
    require(record.get('command') == controller['command'] and int(record['index']) == ticket['start']
            and int(record['count']) == ticket['count'], 'producer command or range differs')
    # Force the existing exact receipt/intent/ledger verifier to read originals.
    allocation = record.get('allocation'); receipt_record = deepcopy(record); receipt_record.pop('allocation', None)
    require(adopt_receipt(root, receipt_record) and receipt_record['allocation'] == allocation,
            'producer lacks exact submitted dispatch receipt and ledger')
    job = tickets._owned(root, allocation['job_dir']); environment = read_env(job/'job.env')
    require(environment.get('ISPATIALGYM_TRAINING_DIVERSITY_TICKET') == ticket_ref['path'] and
            environment.get('ISPATIALGYM_TRAINING_DIVERSITY_TICKET_SHA256') == ticket_ref['sha256'] and
            int(environment['START']) == ticket['start'] and int(environment['N']) == ticket['count'] and
            file_digest(tickets._owned(root, environment['ISPATIALGYM_CAMPAIGN_CONFIG'])) == ticket['config_sha256'],
            'executed job differs from ticket/config/range')
    config = json.loads(Path(environment['ISPATIALGYM_CAMPAIGN_CONFIG']).read_text())
    require(config.get('purpose') == 'training_production' and config.get('split') == 'train', 'not exact production execution')
    release_ref = {'path': environment['ISPATIALGYM_TRAINING_RELEASE'],
                   'sha256': environment['ISPATIALGYM_TRAINING_RELEASE_SHA256']}
    release = tickets._read_reference(root, release_ref)
    require(release['config_sha256'] == ticket['config_sha256'], 'executed release config differs')
    closed = queue_closures(record, allocation, root)
    require(set(record.get('groups', {})) == set(closed), 'producer has unresolved or unknown closed groups')
    validation, validation_ref, validation_original = _copy_json(root, root/'validation/controller.json', directory/'validation.json')
    _fresh(validation.get('updated_at'), now)
    entry = validation.get('gates', {}).get(str(allocation['jobid']))
    require(entry and Path(entry['source_job_dir']).resolve() == job and
            str(entry['source_job_id']) == str(allocation['jobid']), 'exact replay observer entry missing')
    require(entry.get('status') in ('validated', 'demo_failed', 'demo_missing', 'replay_failed', 'replay_missing'),
            'replay observation is not terminal')
    retry_path = root/'validation/replay_retry_requests.json'
    retry_original = reference(retry_path) if retry_path.exists() else None
    retries = json.loads(retry_path.read_text()).get('requests', {}) if retry_path.exists() else {}
    retry = retries.get(str(allocation['jobid']))
    require(not retry or retry.get('request_id') in entry.get('applied_retry_requests', []), 'unapplied replay retry pending')
    source_backend = 'vast' if (root/'vast/state/jobs'/(str(allocation['jobid'])+'.json')).exists() else 'cluster'
    jobs = [{'backend': source_backend, 'job_id': str(allocation['jobid']), 'stage': stage} for stage in ('native', 'collection')]
    for attempt in [*entry.get('attempt_history', []), entry]:
        if attempt.get('replay_job_id'):
            jobs.append({'backend': attempt.get('replay_backend', 'cluster'), 'job_id': str(attempt['replay_job_id']), 'stage': 'replay'})
        migration = attempt.get('vast_migration', {})
        if migration.get('original_replay_job_id'):
            jobs.append({'backend': 'cluster', 'job_id': str(migration['original_replay_job_id']), 'stage': 'replay'})
    unique = {(j['backend'], j['job_id'], j['stage']): j for j in jobs}; jobs = list(unique.values())
    require(not record.get('render_jobs') and all(j in jobs for j in controller.get('jobs', [])),
            'additional owned execution stages require their exact observer adapter')
    committed = tickets._committed_ticket_records(root, ticket, ticket_ref)
    by_index = {tickets._index(r): r for r in committed}
    published = {}
    for path in (job/'inst').rglob('*-tro_state.json'):
        raw = json.loads(path.read_text()); meta = raw['ispatialgym']; iid = meta['instance_id']; index = int(iid.rsplit('__', 1)[-1])
        require(ticket['start'] <= index < ticket['start']+ticket['count'] and index not in published, 'TRO outside exact unique range')
        block = meta['campaign']
        require(block['validation_key'] == config['validation_key'] and block['class_id'] == ticket['class_id']
                and block['task'] == ticket['task'] and block['goal_group_id'] in closed, 'TRO outside exact closed configuration')
        published[index] = (iid, path, block['goal_group_id'])
    require(sum(e['written'] for e in closed.values()) == len(published), 'durable group closure/TRO inventory differs')
    require(set(by_index) <= set(published), 'committed position lacks original TRO')
    reviews = {}; audit_refs = [producer_ref, validation_ref]
    for gid, outcome in record.get('groups', {}).items():
        actual = {r['instance_id']: r['split'] for r in committed if r['group_id'] == gid}
        require(outcome.get('accepted') == actual, 'producer claim differs from canonical acceptance')
        require(not actual or outcome.get('review_sha256'), 'committed output lacks original strict review')
        if outcome.get('review_sha256'):
            rp = root/'groups'/gid/'release_review.json'; review = json.loads(rp.read_text())
            require(digest(review) == outcome['review_sha256'], 'producer strict review changed')
            fresh = audit_job(job, review['gate']['path'], root=root,
                              started_at=float(release['campaign_started_at']), group_closed=True, group_id=gid)
            require(digest(fresh) == digest(review), 'strict source evidence changed after claim')
            reviews[gid] = review; audit_refs.append(reference(rp))
    outcomes = {}
    for index in range(ticket['start'], ticket['start']+ticket['count']):
        if index in by_index: continue
        reason = 'native_failed'; detail = 'Terminal native execution published no member at this reserved position.'
        refs = []
        if index in published:
            iid, tro, gid = published[index]; refs.append(reference(tro)); mp = job/'demo'/iid/'meta.json'
            report = json.loads(mp.read_text()) if mp.exists() else None
            if mp.exists(): refs.append(reference(mp))
            reason = 'demo_failed'; detail = 'Published member has no successful completed hard-QA demo.'
            if report and report.get('success') is True:
                try: report_passed(report, 'oracle')
                except ValueError: pass
                else:
                    review = reviews.get(gid, {})
                    duplicates = {r['instance_id'] for r in review.get('excluded_duplicate_members', [])}
                    if iid in duplicates:
                        reason = 'acceptance_rejected'; detail = 'Strict review excluded a duplicate starting configuration.'
                    elif entry['status'] in ('replay_failed', 'replay_missing'):
                        reason = 'replay_failed'; detail = 'Owned completed replay failed or has no durable report; diagnosis required.'
                    else:
                        require(False, 'successful uncommitted member still awaits acceptance reconciliation')
        outcomes[str(index)] = {'status': reason, 'original_artifacts': refs, 'diagnosis': detail,
            'next_action': 'Producer owner '+controller['owner']+': inspect exact '+reason+' evidence and prepare a changed bounded correction before retry.'}
    audit = {'schema': 'isg-training-ticket-producer-audit-v1', 'campaign_id': tickets.CAMPAIGN,
        'observed_at': now, 'ticket': ticket_ref, 'producer': producer_ref, 'validation': validation_ref,
        'original_sources': [producer_original, validation_original, *([retry_original] if retry_original else []),
            controller['queue'], reference(job/'job.env'), release_ref,
            *[receipt_record['allocation_receipt_observation'][k] for k in ('receipt', 'intent', 'ledger')]],
        'queue': controller['queue'], 'dispatch': receipt_record['allocation_receipt_observation'],
        'jobs': jobs, 'unused_outcomes': outcomes, 'committed_records': committed,
        'dispatch_authorization': False}
    ap = directory/'producer_audit.json'; write_json_atomic(ap, audit); ar = reference(ap)
    for outcome in outcomes.values(): outcome['artifacts'] = [ar]
    finalized = {**controller, 'phase': 'finalized', 'observed_at': now, 'jobs': jobs,
        'pending_dispatch': [], 'pending_acceptance': [], 'outcome_artifacts': [ar, *audit_refs],
        'committed_records': {r['instance_id']: digest(r) for r in committed}, 'unused_outcomes': outcomes,
        'next_action': 'Retire only after fresh actual terminal job proofs; preserve committed output and diagnose unused positions.'}
    return finalized, before, audit['original_sources']


def reconcile(root, registration_ref, *, now=None):
    root = Path(root).resolve(); now = time.time() if now is None else now
    registration = tickets._read_reference(root, registration_ref); ticket = tickets._read_reference(root, registration['ticket'])
    directory = root/'controllers/training_ticket_observer'; request = ticket['request_id']
    with atomic_directory_lock(directory/'locks'/(request+'.lockdir')):
        old = tickets._read_state(root).get('retirements', {}).get(request)
        if old: return {'status': 'already_retired', 'receipt': old}
        work = directory/'observations'/request/str(time.time_ns())
        candidate, prior_controller, sources = derive(root, registration, work, now=now)
        candidate_path = work/'candidate_controller.json'; write_json_atomic(candidate_path, candidate)
        terminal = tickets.observe_ticket_jobs(root, reference(candidate_path), work/'backend')
        tickets._terminal_jobs(root, candidate, tickets._read_reference(root, terminal), time.time())
        # Recheck original mutable producers after slow I/O/backend observations.
        for source in [prior_controller, *sources]:
            require(reference(tickets._owned(root, source['path'])) == source, 'original source changed before finalization')
        cp = Path(prior_controller['path']); write_json_atomic(cp, candidate)
        receipt = tickets.retire_ticket(root, registration['ticket'], reference(cp), terminal)
        return {'status': 'retired', 'receipt': receipt}


def cycle(root):
    root = Path(root).resolve(); directory = root/'controllers/training_ticket_observer'; rows = []
    for path in sorted((directory/'registrations').glob('*.json')):
        registration_ref = reference(path)
        try: result = reconcile(root, registration_ref)
        except (OSError, ValueError, KeyError, TypeError) as error:
            result = {'status': 'awaiting_evidence_or_owned_repair', 'reason': str(error),
                      'next_action': 'Registered producer owner: reconcile the named exact missing or changed evidence; observer will recheck without dispatch.'}
        rows.append({'registration': registration_ref, 'observed_at': time.time(), **result})
    report = {'schema': 'isg-training-ticket-observer-v1', 'campaign_id': tickets.CAMPAIGN,
              'observed_at': time.time(), 'rows': rows, 'jobs_modified': False, 'instances_accepted': False}
    write_json_atomic(directory/'status.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--loop', action='store_true'); args = parser.parse_args()
    while True:
        cycle(args.root)
        if not args.loop: return
        time.sleep(30)


if __name__ == '__main__': main()
