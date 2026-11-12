"""Read-only projection of explicit campaign ownership and scheduler evidence.

The existing capacity monitor is the sole caller/writer. This module cannot
submit jobs, change acceptance, or infer ownership from scheduler job names.
"""
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

SCHEMA = 'isg-monitor-status-v1'
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL'}


def epoch(value):
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return dt.datetime.fromisoformat(str(value).replace('Z', '+00:00')).timestamp()
    except (ValueError, TypeError):
        return None


def freshness(observed_at, now, limit=180):
    value = epoch(observed_at)
    return 'missing' if value is None else ('stale' if now-value > limit else 'fresh')


def read_source(path, errors):
    path = Path(path)
    try:
        raw = path.read_bytes()
        return json.loads(raw), {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'file_modified_at': path.stat().st_mtime}
    except (OSError, ValueError) as exc:
        errors.append({'kind': 'observation_failure', 'source': str(path), 'owner': '/root', 'detail': type(exc).__name__})
        return {}, {'path': str(path), 'observation_status': 'missing'}


def accounting(job_ids):
    if not job_ids:
        return []
    text = subprocess.check_output(['sacct', '-nP', '-X', '-j', ','.join(sorted(job_ids)),
        '--format=JobIDRaw,JobID,State,ExitCode,ElapsedRaw,Start,End,AllocTRES'], text=True, timeout=25)
    rows = []
    for line in text.splitlines():
        fields = line.split('|')
        if len(fields) >= 8:
            raw, jid, state, code, elapsed, start, end, tres = fields[:8]
            rows.append({'id': jid, 'raw_id': raw, 'state': state.split()[0].split('+')[0],
                         'exit_code': code, 'elapsed_seconds': int(elapsed or 0),
                         'started_at': start, 'ended_at': end, 'allocated_resources': tres})
    return rows


def project(root, live_jobs, capacity, now=None, accountant=accounting):
    now = now or dt.datetime.now(dt.timezone.utc).timestamp()
    errors, metadata, evidence = [], {}, []
    registry, registry_ref = read_source(Path(root)/'infra/coordination/ownership.json', errors)
    evidence.append(registry_ref)
    for spec in registry.get('job_sources', []):
        source, ref = read_source(spec['path'], errors)
        jid = source.get(spec.get('job_id_field', 'job_id'))
        if not jid:
            errors.append({'kind': 'metadata_missing', 'source': spec['path'], 'owner': '/root', 'detail': 'Submission has no job ID'})
            continue
        ids = [str(jid)+'_'+str(i) for i in spec['array_indices']] if 'array_indices' in spec else [str(jid)]
        for index, identifier in enumerate(ids):
            metadata[identifier] = {**spec, 'job_id': identifier, 'evidence': [ref, registry_ref],
                                    'array_job_id': str(jid) if 'array_indices' in spec else None,
                                    'submitted_at': source.get('submitted_at', source.get('observed_at'))}
    for spec in registry.get('jobs', []):
        metadata[str(spec['job_id'])] = {**spec, 'evidence': spec.get('evidence', [])+[registry_ref]}
    # Root dispatchers publish exact source-receipt paths, so newly admitted
    # work is mapped automatically without a second manual classification list.
    for account in ('minyeong','shiyuc'):
        status_path=Path(root)/'infra/coordinator_dispatch'/('status_'+account+'.json')
        if not status_path.exists():continue
        status,status_ref=read_source(status_path,errors)
        for item in status.get('requests',[]):
            receipt_path=item.get('submission_path')
            if not receipt_path or not Path(receipt_path).exists():continue
            receipt,receipt_ref=read_source(receipt_path,errors)
            if receipt.get('job_id'):
                jid=str(receipt['job_id'])
                metadata[jid]={key:receipt.get(key) for key in ('workstream','stage','owner','class_id','physical_dedup_key')}
                metadata[jid].update(job_id=jid,path=receipt_path,account=account,
                    submitted_at=receipt.get('submitted_at',receipt.get('observed_at')),evidence=[receipt_ref,status_ref])
    followthrough=Path(root)/'infra/coordinator_dispatch/recovery_member_followthrough.json'
    if followthrough.exists():
        followed,followed_ref=read_source(followthrough,errors)
        for result in followed.get('parent_results',[]):
            for ref in result.get('prepared',[]):
                prepared_path=Path(ref['path']);receipt_path=prepared_path.parent/'submitted.json'
                if not receipt_path.exists():continue
                receipt,receipt_ref=read_source(receipt_path,errors)
                if not receipt.get('job_id'):continue
                cfg,cfg_ref=read_source(receipt['configuration']['path'],errors)
                jid=str(receipt['job_id']);metadata[jid]={'job_id':jid,'path':str(receipt_path),
                    'workstream':'canonical recovery','stage':'demo','owner':'/root/dataset_contract',
                    'class_id':cfg.get('template',{}).get('class_id'),'account':'minyeong',
                    'evidence':[receipt_ref,followed_ref,cfg_ref]}
    # Canonical replay ownership is explicit in controller gates; class identity
    # comes from the gate key, never from the scheduler name.
    validation, validation_ref = read_source(Path(root)/'validation/controller.json', errors)
    for gate in validation.get('gates', {}).values():
        for key, stage in [('source_job_id', 'demo'), ('replay_job_id', 'replay')]:
            jid = str(gate.get(key) or '')
            if jid and jid not in metadata:
                metadata[jid] = {'job_id': jid, 'class_id': gate.get('key'), 'stage': stage,
                    'workstream': 'canonical recovery', 'owner': registry.get('canonical_owner'),
                    'evidence': [validation_ref, registry_ref], 'gate_status': gate.get('status')}
    # Query current owned jobs and recent explicit registrations, not all historical gates.
    tracked = {str(j['id']) for j in live_jobs} | {str(j['job_id']) for j in registry.get('jobs', [])}
    tracked |= {jid for jid, m in metadata.items() if 'path' in m}
    tracked |= {str(j['job_id']) for j in registry.get('controllers', []) if j.get('job_id')}
    scheduled = {str(j['id']): {**j, 'execution_observed_at': now} for j in live_jobs}
    try:
        for row in accountant(tracked):
            scheduled.setdefault(row['id'], {**row, 'execution_observed_at': now})
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        errors.append({'kind': 'observation_failure', 'source': 'sacct', 'owner': '/root', 'detail': type(exc).__name__})
    rows = []
    for jid in sorted(tracked):
        observed, meta = scheduled.get(jid, {}), metadata.get(jid, {})
        mapped = bool(meta and meta.get('owner') and meta.get('workstream') and meta.get('stage'))
        rows.append({**observed, **meta, 'job_id': jid, 'backend': observed.get('backend', 'cluster'),
            'account': observed.get('user', meta.get('account')),
            'mapping_status': 'mapped' if mapped else 'missing', 'mapping_remediation_owner': None if mapped else '/root',
            'execution_state': observed.get('state', 'UNKNOWN'), 'execution_observed_at': observed.get('execution_observed_at'),
            'observation_status': freshness(observed.get('execution_observed_at'), now),
            'utilization_percent': None, 'last_useful_progress_at': meta.get('last_useful_progress_at')})
    by_id = {row['job_id']: row for row in rows}
    backlog = []
    for original in registry.get('backlog', []):
        row = dict(original)
        row['observation_status'] = freshness(row.get('last_verified_event_at'), now, row.get('review_interval_seconds', 900))
        linked = [by_id[jid] for jid in row.get('jobs', []) if jid in by_id]
        row['execution_states'] = [{'job_id': j['job_id'], 'execution_state': j['execution_state'], 'execution_observed_at': j['execution_observed_at']} for j in linked]
        if row.get('stage') == 'ready_for_dispatch' and not row.get('jobs'):
            errors.append({'kind': 'ready_not_submitted', 'requirement': row['requirement'], 'owner': row.get('dispatch_owner', '/root'), 'ready_since': row.get('ready_since'), 'waiting_seconds': max(0, now-epoch(row['ready_since'])) if epoch(row.get('ready_since')) else None})
        if linked and all(j['execution_state'] in TERMINAL for j in linked) and not row.get('terminal_review'):
            errors.append({'kind': 'completed_not_advanced', 'requirement': row['requirement'], 'owner': row.get('owner'), 'jobs': [j['job_id'] for j in linked]})
        if row['observation_status'] != 'fresh':
            errors.append({'kind': 'stale_source', 'requirement': row['requirement'], 'owner': row.get('owner'), 'last_verified_event_at': row.get('last_verified_event_at')})
        if row.get('stage') == 'failed' and not row.get('owner'):
            errors.append({'kind': 'unowned_failure', 'requirement': row['requirement'], 'owner': '/root'})
        for job in linked:
            expected = row.get('expected_duration_seconds')
            if expected and job['execution_state'] == 'RUNNING' and job.get('elapsed_seconds', 0) > expected:
                errors.append({'kind': 'expected_duration_exceeded', 'requirement': row['requirement'], 'owner': row.get('owner'), 'job_id': job['job_id'], 'detail': 'Elapsed exceeds declared bound; useful progress must be inspected before declaring stall'})
        backlog.append(row)
    workers = []
    for spec in registry.get('controllers', []):
        job = by_id.get(str(spec.get('job_id')), {})
        worker = {**spec, 'kind': 'persistent_controller', 'process_state': 'UNKNOWN',
                  'process_observed_at': None, 'last_reconciliation_at': None, 'last_useful_progress_at': None,
                  'allocation': {'state':job.get('execution_state','UNKNOWN'), 'execution_observed_at':job.get('execution_observed_at')}}
        if spec.get('report'):
            report, ref = read_source(spec['report'], errors)
            worker['evidence'] = [ref, registry_ref]
            worker['last_reconciliation_at'] = report.get(spec.get('reconciliation_field', 'observed_at'))
            worker['last_useful_progress_at'] = report.get(spec.get('progress_field', 'last_useful_progress_at'))
        worker['observation_status'] = freshness(worker['last_reconciliation_at'], now, spec.get('stale_after_seconds', 300))
        workers.append(worker)
    return {'schema': SCHEMA, 'schema_version': 1, 'campaign_id': 'campaign_20261110', 'observed_at': now,
            'owner': 'existing local campaign monitor', 'cadence_seconds': 45, 'jobs_observed_at': now, 'jobs': rows, 'workers': workers,
            'backlog': backlog, 'exceptions': errors, 'capacity': capacity, 'capacity_blockers': registry.get('capacity_blockers', []),
            'evidence': evidence, 'utilization_status': 'unmeasured'}


def publish(root, live_jobs, capacity):
    data = project(root, live_jobs, capacity)
    path = Path(root)/'infra/monitor_status.json'
    temporary = path.with_suffix('.json.new')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    temporary.replace(path)
    return data
