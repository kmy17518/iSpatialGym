"""Exact, restartable migration for bounded standalone queue observers.

Call under the observer's state lock, before terminal-status skips. Running
allocations are never touched; source configuration and physical plans remain
byte-identical apart from fresh group/index execution identities.
"""
from copy import deepcopy
from pathlib import Path
import getpass
import json
import time

from scripts.campaign_20261110.contracts import file_digest, require, write_json_atomic
from scripts.campaign_20261110 import controller_migration as boundary
from scripts.campaign_20261110.controller_families import reserve_indices
from scripts.campaign_20261110.controller_spool import enqueue_record
from scripts.campaign_20261110.infra_dispatch_migration import environment, set_environment, immutable_bytes
from scripts.campaign_20261110.infra_dispatch import set_option


def restore_routing(root, replacement):
    """Resume an old prepared cancellation without widening original routes."""
    if not replacement.get('migration_routing'):
        from scripts.campaign_20261110.migration_routes import source_routes
        source = replacement['pending_migration_source']
        require(file_digest(source['record']) == source['sha256'], 'saved migration source record changed')
        routing = source_routes(root, json.loads(Path(source['record']).read_text()))
        require(routing['source_intent'] is not None, 'queue migration needs immutable source dispatch intent')
        replacement['migration_routing'] = routing
        replacement['dispatch_backends'] = routing['backends']
    proof = replacement['pending_migration_source'].get('proof')
    if proof:
        replacement['migration_routing']['proof'] = proof


def prepare_replacement(root, record):
    from ispatialgym.sampling.campaign_goal_queue import read_queue
    root = Path(root).resolve()
    source = Path(record['queue']).resolve()
    require(source.is_relative_to(root/'queues'), 'migration queue outside campaign')
    raw = json.loads(source.read_text())
    config = Path(raw['config_path']).resolve()
    require(config.is_relative_to(root/'groups'), 'migration config outside campaign')
    read_queue(source, config)
    require(record['validation_key'] == json.loads(config.read_text())['validation_key'], 'queue validation key changed')
    require(not record.get('groups'), 'cannot migrate a queue with claimed groups')
    require(not any(p.is_file() and p.stat().st_size for p in source.parent.glob('*.jsonl')), 'queue has native execution events')
    from scripts.campaign_20261110.migration_routes import source_routes
    routing = source_routes(root, record)
    require(routing['source_intent'] is not None, 'queue migration needs immutable source dispatch intent')
    command = list(record['command']); env = environment(command)
    require(Path(env['ISPATIALGYM_CAMPAIGN_GOAL_QUEUE']).resolve() == source, 'command queue differs')
    require(file_digest(env['ISPATIALGYM_CAMPAIGN_CONFIG']) == file_digest(config), 'command config differs')
    token = source.parent.name + '_pending_m01'
    directory = root/'queues'/token
    directory.mkdir(parents=True, exist_ok=True)
    source_record = directory/'migration_source_record.json'
    if not source_record.exists(): write_json_atomic(source_record, deepcopy(record))
    source_hashes = {str(source): file_digest(source), str(config): file_digest(config)}
    entries, ids = [], []
    for ordinal, entry in enumerate(raw['entries']):
        path = Path(entry['path']).resolve()
        require(path.is_relative_to(root/'groups') and file_digest(path) == entry['sha256'], 'source plan changed')
        plan = json.loads(path.read_text())
        require(plan['group_id'] == raw['group_ids'][ordinal], 'source group identity differs')
        target_dir = root/'groups'/(plan['group_id']+'_pending_m01')
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in ('campaign_config.json', 'source.json', 'contracts.json', 'goal_proposal.json', 'empty_eval_index.json'):
            old = path.parent/name
            if old.exists(): immutable_bytes(old, target_dir/name)
        require(file_digest(target_dir/'campaign_config.json') == file_digest(config), 'replacement config differs')
        new_plan = dict(plan, group_id=target_dir.name)
        target = target_dir/'goal_plan.json'
        if target.exists(): require(json.loads(target.read_text()) == new_plan, 'prepared replacement plan differs')
        else: write_json_atomic(target, new_plan)
        source_hashes[str(path)] = entry['sha256']
        entries.append({'path': str(target), 'sha256': file_digest(target)})
        ids.append(target_dir.name)
    target_config = Path(entries[0]['path']).parent/'campaign_config.json'
    new_raw = dict(raw, entries=entries, group_ids=ids, config_path=str(target_config))
    target_queue = directory/'queue.json'
    if target_queue.exists(): require(json.loads(target_queue.read_text()) == new_raw, 'prepared replacement queue differs')
    else: write_json_atomic(target_queue, new_raw)
    read_queue(target_queue, target_config)
    index = reserve_indices(root, token, int(record['count']))
    set_option(command, '--base-index', index)
    set_environment(command, 'ISPATIALGYM_CAMPAIGN_GOAL_QUEUE', target_queue)
    set_environment(command, 'ISPATIALGYM_CAMPAIGN_CONFIG', target_config)
    new_index = target_config.parent/'empty_eval_index.json'
    immutable_bytes(env['EVAL_INDEX'], new_index)
    set_environment(command, 'EVAL_INDEX', new_index)
    set_environment(command, 'ISGF_SAMPLING_CACHE', root/'cache'/token)
    value = deepcopy(record)
    for name in ('allocation', 'dispatch_intent', 'dispatch_receipt', 'migration', 'migration_intent', 'migration_replacement', 'error'):
        value.pop(name, None)
    value.update(directory=str(directory), queue=str(target_queue), index=index, command=command,
                 groups={}, status='ready', dispatch_backends=routing['backends'], migration_routing=routing)
    value['prepared'] = dict(record['prepared'], queue=str(target_queue), config=str(target_config), group_ids=ids)
    value['pending_migration_source'] = {'record': str(source_record), 'sha256': file_digest(source_record),
        'source_hashes': source_hashes, 'original_job_id': record['allocation']['jobid'],
        'changed_fields': ['group_id', 'queue/config paths', 'globally reserved instance indices', 'sampling cache path']}
    write_json_atomic(directory/'migration_prepared.json', value)
    return value


def migrate_pending_queue(root, record, save, scheduler, *, owner=None, minimum_age=0):
    """Check at most once/minute; recover prepared cancellation/publish boundaries."""
    boundary.overflow_readiness.pending_delay(minimum_age)
    root = Path(root).resolve(); owner = owner or getpass.getuser()
    if record.get('status') == 'ready' and record.get('pending_migration_source', {}).get('proof'):
        proof_ref = record['pending_migration_source']['proof']
        require(file_digest(proof_ref['path']) == proof_ref['sha256'], 'migration completion proof changed')
        restore_routing(root, record)
        save()
        enqueue_record(root, record, save, priority=1, backends=record['dispatch_backends'])
        return True
    allocation = record.get('allocation') or {}; jid = str(allocation.get('jobid', ''))
    if not jid.isdigit() or (root/'vast/state/jobs'/(jid+'.json')).exists(): return False
    recovering = bool(record.get('migration_replacement'))
    if not recovering:
        queued = scheduler.get(jid)
        if record.get('status') in ('complete', 'failed') or not queued or queued[1] != 'PENDING' or queued[2] != owner:
            return False
        now = time.time()
        if now - record.get('queue_migration_last_checked', 0) < 60: return False
        record['queue_migration_last_checked'] = now; save()
        proof = boundary.scheduler_proof(jid)
        if not boundary.eligible(proof, owner=owner, minimum_age=minimum_age, now=now): return False
        boundary.no_execution_artifacts(allocation['job_dir'], root)
        record['migration_replacement'] = prepare_replacement(root, record)
        save()  # Fresh immutable identities exist before any scheduler mutation.
    replacement = record['migration_replacement']
    restore_routing(root, replacement)
    save()
    for path, digest in replacement['pending_migration_source']['source_hashes'].items():
        require(file_digest(path) == digest, 'source input changed before migration')
    proof = record.get('migration') or boundary.migrate_record(root, record, minimum_age=minimum_age,
        owner=owner, save=save, execute=True)
    if not proof:
        after = boundary.scheduler_proof(jid)
        if after['scheduler']['JobState'] != 'PENDING':
            record['queue_migration_aborted'] = {'reason': 'original started before migration', 'proof': after,
                'unused_preparation': replacement['directory']}
            record.pop('migration_replacement', None); record.pop('migration_intent', None); save()
        return False
    require(proof.get('never_started') and boundary.confirmed_never_started(proof['after']), 'never-started cancellation not proved')
    proof_path = Path(replacement['directory'])/'never_started_proof.json'
    if proof_path.exists(): require(json.loads(proof_path.read_text()) == proof, 'migration proof changed')
    else: write_json_atomic(proof_path, proof)
    replacement = deepcopy(replacement)
    replacement['pending_migration_source']['proof'] = {'path': str(proof_path), 'sha256': file_digest(proof_path)}
    replacement['migration_routing']['proof'] = replacement['pending_migration_source']['proof']
    # Re-run the existing exact dry-run on the new execution identities. The
    # original native code, config bytes and physical goal fields are unchanged.
    from scripts.campaign_20261110.controller_families import run_logged
    run_logged(replacement['command']+['--dry-run'], Path(replacement['directory'])/'migration_dryrun.log')
    record.clear(); record.update(replacement); save()
    enqueue_record(root, record, save, priority=1, backends=record['dispatch_backends'])
    return True
