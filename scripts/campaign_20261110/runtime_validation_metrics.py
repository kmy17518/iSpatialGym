"""Scoped, read-only cost evidence for one frozen current-runtime class suite.

Registration precedes enqueue. Native and collector processes overlap on one
GPU: process timings are diagnostic components, never additive allocation cost.
The acceptance owner supplies strict outcomes; this reader cannot grant credit.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

CAMPAIGN = 'campaign_20261110'
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def reference(path):
    path = Path(path)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def checked(ref):
    require(reference(ref['path'])['sha256'] == ref['sha256'], 'evidence hash differs: ' + ref['path'])
    return json.loads(Path(ref['path']).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n'); temporary.replace(path)


def validate_row(row):
    require(row.get('run_id') and row.get('class_id'), 'suite/class identity missing')
    require(row.get('run_started_at') is not None, 'suite start missing')
    runtime = row['frozen_runtime']; native = Path(runtime['path'])
    commit_file = native / '.snapshot_commit'
    require(reference(commit_file)['sha256'] == runtime['snapshot_sha256'], 'frozen runtime hash differs')
    require(commit_file.read_text().strip() == runtime['commit'], 'frozen runtime commit differs')
    config = checked(row['campaign_config'])
    require(config['campaign_id'] == CAMPAIGN and config['template']['class_id'] == row['class_id'], 'config class/campaign differs')
    require(config['validation_key']['code_sha256'] == runtime['snapshot_sha256'], 'config is not pinned to suite runtime')
    require(type(row['fresh_start']) is int and type(row['fresh_count']) is int and 1 <= row['fresh_count'] <= 3, 'invalid bounded fresh range')
    camera = row.get('camera_exempt') is True
    require(camera == config['template']['task'].startswith('camera_'), 'camera exemption differs from native task')
    if camera:
        require(not row.get('group_id') and not row.get('goal_plan'), 'camera cannot have fabricated group identity')
    else:
        plan = checked(row['goal_plan'])
        require(plan['group_id'] == row['group_id'], 'goal group identity differs')
        require(plan.get('max_members') == 3, 'goal plan must retain maximum three members')
    return config


def register(row_path):
    """Validate one immutable row and retain a pre-dispatch measurement receipt."""
    row_path = Path(row_path); row = json.loads(row_path.read_text()); validate_row(row)
    from .runtime_suite_paths import base_for_row, paths
    base=base_for_row(row_path);target=paths(base,row_path)['registration']
    target.parent.mkdir(parents=True,exist_ok=True)
    require(row_path.parent.name!='rows' or row_path.stem==row['class_id'],'row locator differs')
    value = {'schema': 'isg-runtime-validation-measurement-registration-v1', 'campaign_id': CAMPAIGN,
             'run_id': row['run_id'], 'class_id': row['class_id'], 'row': reference(row_path),
             'registered_at': time.time(), 'native_instrumentation': ['campaign_attempts.jsonl',
             'campaign_sampling_failures.jsonl', 'job.txt', 'job_stats.json', 'demo/*/meta.json'],
             'cost_basis': 'unique GPU allocations; overlapping native/collector timing reported separately',
             'acceptance_authority': False}
    try:
        with target.open('x') as stream:
            json.dump(value, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    except FileExistsError:
        prior = json.loads(target.read_text())
        require(prior['row'] == value['row'], 'registered suite row changed')
    return reference(target)


def attempts(events, row):
    """Exact native draw starts, including unfinished attempts; no slot inference."""
    started, finished = {}, {}
    for event in events:
        require(event['class_id'] == row['class_id'], 'attempt class differs')
        require(event['config_sha256'] == row['campaign_config']['sha256'], 'attempt config differs')
        require(event['validation_key']['code_sha256'] == row['frozen_runtime']['snapshot_sha256'], 'attempt runtime differs')
        require(row['fresh_start'] <= event['index'] < row['fresh_start'] + row['fresh_count'], 'attempt outside registered range')
        require(event.get('goal_group_id') == row.get('group_id'), 'attempt group differs')
        destination = started if event['event'] == 'started' else finished
        require(event['event'] in ('started', 'finished'), 'unknown attempt event')
        key = event['attempt_id']
        require(key not in destination or destination[key] == event, 'conflicting duplicate native event')
        destination[key] = event
    durations, failed = [], []
    for key, end in finished.items():
        require(key in started, 'finished native attempt has no durable start')
        elapsed = end['created_at'] - started[key]['created_at']; require(elapsed >= 0, 'negative attempt duration')
        durations.append(elapsed)
        if end.get('outcome') == 'rejected': failed.append(elapsed)
    return {'attempted_instance_slots': len({e['index'] for e in started.values()}),
            'native_draws': len(started), 'native_finished_draws': len(finished),
            'censored_native_draws': len(set(started) - set(finished)),
            'native_process_seconds': sum(durations), 'failed_native_process_seconds': sum(failed),
            'native_draw_elapsed_seconds': durations,
            'native_rejections': dict(Counter(e.get('reason') or 'unspecified' for e in finished.values() if e.get('outcome') == 'rejected')),
            'first_native_started_at': min((e['created_at'] for e in started.values()), default=None),
            'last_native_finished_at': max((e['created_at'] for e in finished.values()), default=None)}


def accounting(rows):
    """Deduplicate allocation identity; do not add phase/process/rental costs."""
    unique = {}
    for row in rows:
        key = (row['backend'], str(row['job_id']))
        require(key not in unique or unique[key] == row, 'conflicting duplicate allocation cost')
        unique[key] = row
    total = completed = censored = 0.0; missing = []
    for key, row in unique.items():
        seconds = row.get('allocated_gpu_seconds')
        if seconds is None: missing.append(list(key)); continue
        require(seconds >= 0, 'negative allocation cost')
        total += seconds
        if row.get('terminal'): completed += seconds
        else: censored += seconds
    return {'observed_allocated_gpu_hours': total / 3600, 'terminal_allocated_gpu_hours': completed / 3600,
            'unfinished_consumed_gpu_hours': censored / 3600, 'cost_complete': not missing,
            'missing_allocation_costs': missing, 'unique_allocations': len(unique),
            'shared_rental_provisioning_idle_gpu_hours': None,
            'shared_rental_cost_note': 'Reported separately at campaign rental level; never added again to per-job GPU time.'}


def rates(cost, outcomes):
    """Eval cost retains sibling production and failed retries in its numerator."""
    keys = ['attempted_instance_slots', 'demo_successful_instances', 'replay_passed_instances',
            'accepted_eval_instances', 'completed_goal_groups']
    result = {}
    for key in keys:
        n = outcomes.get(key)
        result['gpu_hours_per_' + key] = (cost['observed_allocated_gpu_hours'] / n
            if cost['cost_complete'] and isinstance(n, int) and n > 0 else None)
    return result


def scheduler(job_ids):
    if not job_ids: return {}, None
    command = ['sacct', '-n', '-P', '-X', '-j', ','.join(sorted(set(job_ids))),
               '--format=JobID,JobIDRaw,State,ExitCode,ElapsedRaw,Submit,Start,End,AllocTRES']
    proc = subprocess.run(command, capture_output=True, text=True, check=True, timeout=60,
                          env=dict(os.environ, TZ='UTC'))
    result = {}
    for line in proc.stdout.splitlines():
        a = line.split('|'); require(len(a) >= 9, 'invalid scheduler accounting row')
        tres = dict(x.split('=', 1) for x in a[8].split(',') if '=' in x)
        gpu_count = int(tres['gres/gpu']) if 'gres/gpu' in tres else 0 if not a[8] else None
        terminal = a[2].split()[0] in TERMINAL
        result[a[0]] = {'backend': 'slurm', 'job_id': a[0], 'raw_job_id': a[1], 'state': a[2], 'exit_code': a[3],
                          'elapsed_seconds': int(a[4]), 'submitted_at': a[5], 'acquired_at': a[6], 'ended_at': a[7],
                          'allocated_resources': a[8], 'gpu_count': gpu_count, 'terminal': terminal,
                          'allocated_gpu_seconds': int(a[4]) * gpu_count if gpu_count is not None else None}
    return result, {'command': command, 'timezone': 'UTC', 'stdout': proc.stdout, 'observed_at': time.time()}


def vast_accounting(root, job, evidence_dir, now=None):
    """Join actual immutable worker attempts; shared rental startup stays separate."""
    root = Path(root); now = time.time() if now is None else now
    jid = str(job['job_id']); state_path = root/'vast/state/jobs'/(jid+'.json')
    data = state_path.read_bytes(); state = json.loads(data)
    require(str(state['id']) == jid, 'Vast job identity differs')
    transport = Path(state['job_dir']).resolve(); replay_ref = None
    if transport != Path(job['job_dir']).resolve():
        require(job.get('stage') == 'physics_replay' and state.get('job_kind') == 'physics_replay',
                'Vast job identity differs')
        manifest_path = transport/'replay_inputs.json'; replay_ref = reference(manifest_path)
        require(replay_ref['sha256'] == state['replay_inputs_sha256'], 'Vast replay manifest hash differs')
        manifest = checked(replay_ref)
        require(manifest['schema'] == 'isg-vast-physics-replay-inputs-v1' and
                Path(manifest['campaign_root']).resolve() == root.resolve() and
                Path(manifest['job_dir']).resolve() == transport and
                transport.is_relative_to(root.resolve()/'out/vast_replay') and
                Path(manifest['canonical_output']).absolute() == Path(job['job_dir']).absolute() and
                Path(manifest['source_job']).resolve() == Path(state['source_job']).resolve() and
                Path(job['job_dir']).absolute().is_relative_to(root.resolve()/'validation') and
                Path(job['job_dir']).is_symlink() and
                Path(job['job_dir']).resolve() == (transport/'gate').resolve(),
                'Vast replay canonical output binding differs')
    retained = Path(evidence_dir)/(jid+'_'+hashlib.sha256(data).hexdigest()+'.json')
    retained.parent.mkdir(parents=True, exist_ok=True)
    if not retained.exists(): retained.write_bytes(data)
    evidence = [reference(retained)] + ([replay_ref] if replay_ref else []); attempts_by_id = {}; parts = []; missing = []
    for path in sorted((root/'vast/state/attempts'/jid).glob('*.json')):
        item = json.loads(path.read_text()); done = item['done']; aid = item['attempt_id']
        require(item['schema'] == 'isg-vast-attempt-v1' and str(item['job_id']) == jid and
                str(done['id']) == jid and done['attempt_id'] == aid and
                Path(done['job_dir']).resolve() == transport, 'Vast attempt identity differs')
        require(aid not in attempts_by_id, 'duplicate Vast attempt identity')
        require(done.get('gpu_count') == 1, 'multi-GPU rental cost allocation is unavailable')
        begin_path = root/'vast/state/cost_provenance'/('assignment_begin_'+hashlib.sha256(('assignment_begin\0'+aid).encode()).hexdigest()+'.json')
        begin = json.loads(begin_path.read_text())
        require(begin['kind'] == 'assignment_begin' and begin['identity'] == aid and
                str(begin['job_id']) == jid and str(begin['instance']) == str(item['instance']) and
                begin['assignment_started_unix'] == item['assigned_unix'], 'Vast assignment provenance differs')
        require(Path(begin['scope']['job_dir']).resolve() == transport, 'Vast assignment scope differs')
        times = [item['assigned_unix'], done['fetch_started_unix'], done['runtime_started_unix'],
                 done['runtime_ended_unix'], done['upload_finished_unix']]
        require(all(isinstance(t, (int, float)) and not isinstance(t, bool) and 0 < t <= now for t in times)
                and times == sorted(times), 'Vast worker timestamps invalid')
        part = {'attempt_id': aid, 'instance': str(item['instance']), 'terminal': True, 'rc': done.get('rc'),
                'allocated_gpu_seconds': times[-1]-times[0], 'assignment_to_fetch_seconds': times[1]-times[0],
                'fetch_seconds': times[2]-times[1], 'runtime_seconds': times[3]-times[2],
                'upload_seconds': times[4]-times[3], 'assignment': reference(begin_path), 'attempt': reference(path)}
        attempts_by_id[aid] = part; parts.append(part); evidence.extend([part['assignment'], part['attempt']])
    tries = int(state.get('tries', 0)); require(tries >= len(parts), 'Vast try count contradicts preserved attempts')
    terminal = state.get('status') in ('synced', 'failed')
    # A current worker heartbeat supplies a censored lower bound, never a guessed end.
    if state.get('status') == 'assigned' and len(parts) < tries:
        begin = checked(state['cost_assignment_begin']); aid = begin['identity']
        require(str(begin['job_id']) == jid and str(begin['instance']) == str(state['instance']) and
                begin['assignment_started_unix'] == state['assigned'] and
                Path(begin['scope']['job_dir']).resolve() == transport, 'current Vast assignment differs')
        hb_path = root/'vast/worker_heartbeats'/(str(state['instance'])+'.json')
        hb_data = hb_path.read_bytes(); hb = json.loads(hb_data)['heartbeat']; running = hb.get('running', {}).get(jid)
        if aid not in attempts_by_id and running and 0 <= now-hb['time'] <= 180 and hb['time'] >= state['assigned']:
            hb_retained = Path(evidence_dir)/('heartbeat_'+hashlib.sha256(hb_data).hexdigest()+'.json')
            if not hb_retained.exists(): hb_retained.write_bytes(hb_data)
            parts.append({'attempt_id': aid, 'instance': str(state['instance']), 'terminal': False,
                          'allocated_gpu_seconds': hb['time']-state['assigned'],
                          'observed_through': hb['time'], 'phase': running.get('phase'), 'cpus': running.get('cpus'),
                          'assignment': state['cost_assignment_begin'], 'heartbeat': reference(hb_retained)})
            evidence.extend([state['cost_assignment_begin'], reference(hb_retained)])
    if len(parts) != tries: missing.append('Unobserved or lost worker attempts: '+str(tries-len(parts)))
    seconds = sum(p['allocated_gpu_seconds'] for p in parts)
    return {'backend': 'vast', 'job_id': jid, 'state': state.get('status'), 'terminal': terminal,
            'allocated_gpu_seconds': None if missing else seconds, 'known_attempt_gpu_seconds': seconds,
            'attempts': parts, 'missing_cost_evidence': missing, 'evidence': evidence,
            'cost_boundary': 'Assignment through actual upload completion, including failed attempts; runtime is an overlapping component. Shared rental provisioning/idle is separate.'}


def json_lines(path):
    lines = Path(path).read_bytes().splitlines(keepends=True); output = []
    for i, line in enumerate(lines):
        try: output.append(json.loads(line))
        except ValueError:
            if i == len(lines) - 1 and not line.endswith(b'\n'): break
            raise
    return output


def sample_distribution(values):
    values = sorted(values)
    if not values: return {'n': 0, 'min': None, 'median': None, 'max': None}
    middle = len(values) // 2
    return {'n': len(values), 'min': values[0], 'median': (values[middle] if len(values) % 2 else (values[middle-1]+values[middle])/2), 'max': values[-1]}


def same_execution_snapshot(accepted_ref, current_ref, row_ref):
    """A frozen acceptance input may precede later observational/cost updates."""
    if accepted_ref == current_ref:
        return
    accepted = checked(accepted_ref); current = checked(current_ref)
    require(accepted['row'] == current['row'] == row_ref and
            all(accepted[k] == current[k] for k in ('allocation', 'dispatch_receipt')),
            'strict outcome execution reference differs')
    identity = lambda r: tuple(str(r[k]) for k in ('job_id', 'job_dir', 'backend', 'stage'))
    require({identity(r) for r in accepted['records']} <= {identity(r) for r in current['records']},
            'accepted execution jobs disappeared or changed')
    require(all(g in current.get('member_gates', []) for g in accepted.get('member_gates', [])),
            'accepted member replay evidence disappeared or changed')
    require(not accepted.get('gate') or accepted['gate'] == current.get('gate') or
            accepted['gate'] in current.get('member_gates', []), 'accepted gate changed')


def strict_outcomes(result, row, row_ref, allocation_ref):
    """Join the strict owner's exact evidence; regression consumes no quota."""
    require(result.get('schema') == 'isg-current-runtime-class-acceptance-v1', 'unknown strict outcome schema')
    for key in ('run_id', 'class_id', 'frozen_runtime', 'campaign_config', 'group_id', 'fresh_start', 'fresh_count', 'camera_exempt'):
        require(result[key] == row[key], 'strict outcome identity differs: ' + key)
    require(result['row'] == row_ref, 'strict outcome row reference differs')
    same_execution_snapshot(result['allocation'], allocation_ref, row_ref)
    review = checked(result['strict_review']); validation = checked(result['regression_audit'])
    quota = checked(result['quota_evidence'])
    require(quota == result['quota'], 'strict outcome quota differs')
    require(result.get('physical_validation_passed') is True and validation.get('validation_passed') is True, 'physical validation did not pass')
    ids = [m['instance_id'] for m in result['members']]
    require(len(ids) == len(set(ids)) and set(ids) == {m['instance_id'] for m in review['members']}, 'strict member identity differs')
    require(all(m.get('demo_passed') is True and m.get('physics_replay_passed') is True and
                m.get('physics_replay_applicability') == 'own_member_replay' for m in result['members']), 'strict member replay missing')
    require(len(ids) >= (1 if row['camera_exempt'] else 2), 'strict successful member count insufficient')
    if result['status'] == 'regression_validation_passed':
        require(quota['quota_ineligible'] is True and quota['eligible'] is False and quota['consumed'] == 0 and quota['accepted'] == {}, 'regression consumed benchmark quota')
    require(result['status'] in ('accepted', 'regression_validation_passed', 'validated_pending_acceptance'), 'unknown strict outcome status')
    return {'status': result['status'], 'validation_passed': result['validation_passed'],
            'replay_passed_instances': len(ids), 'accepted_eval_instances': sum(v == 'eval_standard' for v in quota['accepted'].values()),
            'accepted_training_instances': sum(v == 'train' for v in quota['accepted'].values()),
            'completed_goal_groups': int(not row['camera_exempt'] and result['validation_passed']),
            'regression_validation_groups': int(not row['camera_exempt'] and result['status'] == 'regression_validation_passed'),
            'camera_validation_instances': len(ids) if row['camera_exempt'] else 0,
            'quota_ineligible': quota['quota_ineligible'], 'source': 'strict acceptance owner; exact linked evidence checked'}


def rollup_attempts(records):
    """Latest exact result with all historical allocation cost retained once."""
    grouped={}
    for record in records:grouped.setdefault(record['class_id'],[]).append(record)
    result=[]
    for rows in grouped.values():
        if len(rows)==1:result.append(rows[0]);continue
        latest=dict(rows[-1]);latest['attempt_history']=rows
        jobs={}
        for row in rows:
            for job in row['jobs']:
                key=(job['backend'],str(job['job_id']))
                require(key not in jobs or jobs[key]==job,'attempts share conflicting allocation evidence')
                jobs[key]=job
        latest['jobs']=list(jobs.values());latest['cost']=accounting(latest['jobs'])
        latest['current_attempt_native_attempts']=latest['native_attempts']
        combined={}
        for key in ('attempted_instance_slots','native_draws','native_finished_draws','censored_native_draws','native_process_seconds','failed_native_process_seconds'):
            combined[key]=sum(row['native_attempts'].get(key,0) for row in rows)
        combined['native_draw_elapsed_seconds']=[x for row in rows for x in row['native_attempts'].get('native_draw_elapsed_seconds',[])]
        counts=Counter()
        for row in rows:counts.update(row['native_attempts'].get('native_rejections',{}))
        combined['native_rejections']=dict(counts);latest['native_attempts']=combined
        latest['distinct_demo_instance_ids']=sorted({iid for row in rows for iid in row['distinct_demo_instance_ids']})
        latest['errors']=[error for row in rows for error in row['errors']]
        if latest['errors']:latest['cost']['cost_complete']=False
        latest['rates']=rates(latest['cost'],{'attempted_instance_slots':combined['attempted_instance_slots'],'demo_successful_instances':len(latest['distinct_demo_instance_ids']),**(latest['strict_outcomes'] or {})})
        latest['cost_scope']='All registered attempts, including failed, canceled and optional replay allocations; unique backend/job counted once.'
        result.append(latest)
    return result


def observe(run_dir):
    """Only explicit suite rows and job directories; no global allocation scans."""
    run_dir = Path(run_dir); records, errors, ids = [], [], []
    from .runtime_suite_paths import all_rows, paths
    for row_path in all_rows(run_dir.parents[1],run_dir):
        try:
            row = json.loads(row_path.read_text()); validate_row(row)
            registration = paths(run_dir,row_path)['registration']
            require(registration.exists() and json.loads(registration.read_text())['row'] == reference(row_path), 'pre-dispatch measurement registration missing')
            path = paths(run_dir,row_path)['allocation']
            allocation = json.loads(path.read_text()) if path.exists() else {'records': []}
            jobs = allocation.get('records', [])
            for job in jobs:
                require(job.get('stage') in ('generation_collection', 'physics_replay', 'rendering'), 'unknown measured allocation stage')
                require(job.get('backend'), 'backend identity missing')
                if job['backend'] != 'vast': ids.append(str(job['job_id']))
            records.append((row_path, row, jobs, reference(path) if path.exists() else None))
        except Exception as exc: errors.append({'row': str(row_path), 'error': type(exc).__name__ + ': ' + str(exc)})
    account, raw = scheduler(ids)
    evidence_dir = run_dir / 'measurements' / 'scheduler'; evidence_dir.mkdir(parents=True, exist_ok=True)
    scheduler_ref = None
    if raw:
        raw_path = evidence_dir / (str(time.time_ns()) + '.json'); write(raw_path, raw); scheduler_ref = reference(raw_path)
    output = []
    for row_path, row, jobs, allocation_ref in records:
        result = {'class_id': row['class_id'], 'row': reference(row_path), 'runtime': row['frozen_runtime'],
                  'camera_exempt': row.get('camera_exempt', False), 'allocation_record': allocation_ref,
                  'observed_at': time.time(), 'evidence': [], 'errors': [], 'jobs': [], 'native_attempts': {},
                  'strict_outcomes': None, 'suggested_time_limits': None,
                  'recommendation_confidence': 'Insufficient completed exact groups; retain registered limits.'}
        events = []; process = []; demo_ids = set()
        for job in jobs:
            try:
                jid = str(job['job_id']); directory = Path(job['job_dir']); measured = account.get(jid)
                if job['backend'] == 'vast':
                    try:
                        measured = vast_accounting(run_dir.parents[1], job, run_dir/'measurements/vast_states')
                    except Exception as exc:
                        measured = {'backend': 'vast', 'job_id': jid, 'terminal': False, 'allocated_gpu_seconds': None,
                                    'state': 'cost_evidence_unavailable', 'error': type(exc).__name__+': '+str(exc)}
                if measured is None:
                    measured = {'backend': job['backend'], 'job_id': jid, 'state': 'unavailable',
                                'terminal': False, 'allocated_gpu_seconds': None}
                result['jobs'].append({**measured, 'stage': job['stage'], 'job_dir': str(directory)})
                if job['stage'] != 'generation_collection': continue
                env = directory / 'job.env'
                from scripts.campaign_20261110.goal_groups import read_env
                environment = read_env(env)
                require(Path(environment['ISGE_CODE']).resolve() == Path(row['frozen_runtime']['path']).resolve(), 'executed runtime differs')
                require(environment.get('ISPATIALGYM_RUNTIME_SUITE_RUN') == row['run_id'], 'execution missing suite provenance')
                result['evidence'].append(reference(env))
                for path in sorted((directory / 'inst').rglob('campaign_attempts.jsonl')):
                    events.extend(json_lines(path)); result['evidence'].append(reference(path))
                stats = directory / 'job_stats.json'
                if stats.exists():
                    st = json.loads(stats.read_text()); process.append({'job_id': jid, **{k:st.get(k) for k in ['sample_wall_s','collector_walls_s','world_load_s','attempts_s','reject_s','demo_meta','job_wall_s']}}); result['evidence'].append(reference(stats))
                for path in sorted((directory / 'demo').glob('*/meta.json')):
                    meta = json.loads(path.read_text())
                    if meta.get('success') is True:
                        from scripts.campaign_20261110.acceptance import report_passed
                        report_passed(meta, 'oracle')
                        require(Path(meta['provenance']['isg_root']).resolve() == Path(row['frozen_runtime']['path']).resolve(), 'demo runtime differs')
                        demo_ids.add(meta['instance_id'])
                    result['evidence'].append(reference(path))
            except Exception as exc: result['errors'].append({'job_id': job.get('job_id'), 'error': type(exc).__name__ + ': ' + str(exc)})
        try: result['native_attempts'] = attempts(events, row)
        except Exception as exc: result['errors'].append({'error': 'native_attempt_identity: ' + str(exc)})
        result['overlapping_process_timings'] = process
        result['cost'] = accounting(result['jobs'])
        if result['errors']:
            result['cost']['cost_complete'] = False
        result['distinct_demo_instance_ids'] = sorted(demo_ids)
        result['native_elapsed_distribution'] = sample_distribution(result['native_attempts'].get('native_draw_elapsed_seconds', []))
        acceptance = run_dir / 'acceptance' / (row['class_id']+'.json')
        if acceptance.exists() and json.loads(acceptance.read_text()).get('row')==reference(row_path):
            result['acceptance_owner_record'] = reference(acceptance)
            try:
                result['strict_outcomes'] = strict_outcomes(json.loads(acceptance.read_text()), row, reference(row_path), allocation_ref)
            except Exception as exc:
                result['errors'].append({'error': 'strict_outcome_identity: ' + str(exc)})
        result['rates'] = rates(result['cost'], {'attempted_instance_slots': result['native_attempts'].get('attempted_instance_slots'), 'demo_successful_instances': len(demo_ids), **(result['strict_outcomes'] or {})})
        output.append(result)
    output=rollup_attempts(output)
    value = {'schema': 'isg-runtime-validation-measurements-v1', 'campaign_id': CAMPAIGN, 'observed_at': time.time(),
             'run_dir': str(run_dir), 'classes': output, 'errors': errors, 'scheduler_evidence': scheduler_ref,
             'scope': 'Exact registered classes only; no acceptance mutations, no historical sample compatibility inference.'}
    write(run_dir / 'measurements/current.json', value)
    lines = ['| Class | Native draws | Censored draws | Demo successes | Observed GPU h | Cost complete | Physical/acceptance status |', '|---|---:|---:|---:|---:|---|---|']
    for r in output:
        n = r['native_attempts']; status = (r['strict_outcomes'] or {}).get('status', 'awaiting strict result'); lines.append(f"| {r['class_id']} | {n.get('native_draws', 'unavailable')} | {n.get('censored_native_draws', 'unavailable')} | {len(r['distinct_demo_instance_ids'])} | {r['cost']['observed_allocated_gpu_hours']:.4f} | {r['cost']['cost_complete']} | {status} |")
    (run_dir / 'measurements/current.md').write_text('\n'.join(lines)+'\n\nNative and collector process times overlap; allocation GPU time is counted once. Missing or censored costs remain explicit.\n')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--register', type=Path)
    parser.add_argument('--run-dir', type=Path); parser.add_argument('--every', type=float, default=0)
    args = parser.parse_args()
    if args.register: print(json.dumps(register(args.register))); return
    require(args.run_dir is not None, '--run-dir required')
    while True:
        result = observe(args.run_dir); print(json.dumps({'classes': len(result['classes']), 'errors': result['errors']}), flush=True)
        if not args.every: return
        time.sleep(args.every)


if __name__ == '__main__': main()
