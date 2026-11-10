"""Read-only goal-group yield from native events, demo reports and committed SQL.

Authored queues, successful demos and accepted instances are different populations.
This observer never invokes a controller, sampler, allocator or job operation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

from scripts.campaign_20261110.acceptance import report_passed
from scripts.campaign_20261110.contracts import Perturbations, ScaleState, digest, require, write_json_atomic
from scripts.campaign_20261110.controller_families import allocations
from scripts.campaign_20261110.export_release import snapshot_allocations
from scripts.campaign_20261110.goal_groups import read_env

CAMPAIGN = 'campaign_20261110'
CONTROL = {'campaign_group_complete', 'campaign_group_timeout', 'campaign_queue_complete'}
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'completed', 'failed', 'done', 'synced'}
MODES = {'eval_standard': 'standard_eval', 'train': 'train_only', 'eval_model': 'held_out', 'eval_category': 'held_out'}


class Evidence:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.sources, self.errors, self.cache = {}, [], {}

    def bytes(self, path, kind):
        path = Path(path).resolve()
        require(path.is_relative_to(self.root), 'yield evidence outside campaign')
        with path.open('rb') as handle:
            import os
            stat = os.fstat(handle.fileno()); raw = handle.read()
        self.sources[str(path)] = {'path': str(path), 'kind': kind, 'sha256': hashlib.sha256(raw).hexdigest(), 'mtime': stat.st_mtime}
        return raw

    def read(self, path, kind='record'):
        key = str(path)
        if key not in self.cache:
            self.cache[key] = json.loads(self.bytes(path, kind))
        return self.cache[key]

    def lines(self, path, kind):
        raw = self.bytes(path, kind)
        result = []
        for i, line in enumerate(raw.splitlines()):
            try: result.append(json.loads(line))
            except ValueError:
                # An append in progress is not a completed record.
                self.errors.append({'source': str(path), 'line': i + 1, 'error': 'incomplete or malformed journal line'})
        return result

    def error(self, path, error):
        self.errors.append({'source': str(path), 'error': type(error).__name__ + ': ' + str(error)})


def member_start(block):
    group = block.get('runtime', {}).get('goal_group', {})
    goal, start = group.get('requested_goal'), group.get('requested_start')
    require(goal and start, 'successful demo lacks intended goal/start provenance')
    p = block['perturbations']
    perturbations = Perturbations({k: ScaleState(**v) for k, v in p['scales'].items()}, tuple(p['removed_objects']), p['lighting'])
    return digest([digest(goal), digest(start), perturbations.fingerprint])


def fresh_group(gid, config, plan, path):
    t = config['template']
    return {'group_id': gid, 'task': t['task'], 'class_id': t['class_id'], 'split': config['split'],
        'mode': MODES[config['split']], 'runtime_revision': 'unobserved',
        'validation_key': config['validation_key'], 'created_at': plan.get('created_at'),
        'plan': str(path), 'job_ids': set(), 'members': {}, 'attempt_slots': set(), 'native_attempts': set(),
        'exact_started': set(), 'exact_slots': set(), 'rejections': Counter(), 'demo_rejections': Counter(),
        'activated': False, 'sampling_closed': False, 'terminal': False, 'closure_sources': set(),
        'pending_execution': False, 'running_execution': False, 'accepted': Counter(), 'accepted_ids': [],
        'acceptance_reasons': [], 'first_attempt_at': None, 'last_attempt_at': None,
        'exact_instrumentation_jobs': set(), 'executed_jobs': set(), 'evidence_incomplete': []}


def controller_outcomes(value, groups, source):
    """Join explicit group identities only, never infer groups from class totals."""
    if isinstance(value, list):
        for row in value: controller_outcomes(row, groups, source)
    elif isinstance(value, dict):
        explicit = value.get('groups')
        if isinstance(explicit, dict):
            for gid,result in explicit.items():
                if gid in groups and isinstance(result,dict):
                    reason=result.get('error') or result.get('reason')
                    if reason:groups[gid]['acceptance_reasons'].append(str(reason))
        if value.get('group_id') in groups and (value.get('error') or value.get('reason')):
            groups[value['group_id']]['acceptance_reasons'].append(str(value.get('error') or value.get('reason')))
        prepared=value.get('prepared') or {}
        if isinstance(prepared,dict) and value.get('error'):
            for gid in prepared.get('group_ids',[]):
                if gid in groups:groups[gid]['acceptance_reasons'].append(str(value['error']))
        for row in value.values():
            if isinstance(row,(dict,list)):controller_outcomes(row,groups,source)


def summarize(groups, jobs):
    """Pure reducer: completed denominator never includes authored or active groups."""
    states = Counter(g['status'] for g in groups)
    completed = [g for g in groups if g['status'] == 'completed']
    hist = Counter(str(g['distinct_demo_successes']) for g in completed)
    accepted = Counter(); reasons = Counter(); outcomes = Counter()
    for g in completed:
        accepted.update(g['accepted']); reasons.update(g['rejections']); reasons.update(g['demo_rejections'])
        reasons.update(g['acceptance_reasons']); outcomes[g['acceptance_outcome']] += 1
    observed = sum(g['observed_attempted_slots'] for g in completed)
    success = sum(g['distinct_demo_successes'] for g in completed)
    complete_denominator = bool(completed) and all(g['attempt_denominator_complete'] for g in completed)
    ids = {jid for g in groups for jid in g['job_ids']}
    cohort_jobs = [j for j in jobs if j['job_id'] in ids]
    terminal_jobs = [j for j in cohort_jobs if j['terminal']]
    # One allocation is counted once even when it loads several groups. Failed
    # setup jobs remain in this cost cohort although they activated zero groups.
    exact_jobs = [j for j in terminal_jobs if j.get('allocated_gpu_seconds') is not None and j['backend'] == 'slurm']
    exact_ids = {j['job_id'] for j in exact_jobs}
    covered_groups = [g for g in completed if g['job_ids'] and set(g['job_ids']) <= exact_ids]
    covered_accepted = sum(sum(g['accepted'].values()) for g in covered_groups)
    hours = sum(j['allocated_gpu_seconds'] for j in exact_jobs) / 3600
    all_hours_known = bool(terminal_jobs) and len(exact_jobs) == len(terminal_jobs)
    all_accepted = sum(accepted.values())
    all_groups_covered = all(set(g['job_ids']) <= exact_ids for g in completed)
    return {'groups_attempted': sum(g['attempted'] for g in groups), 'completed_groups': len(completed),
        'pending_running_unstarted': dict(states), 'success_histogram': {str(i): hist[str(i)] for i in range(4)},
        'more_than_three_successes': sum(n for k,n in hist.items() if int(k)>3),
        'at_least_two_percent': 100*sum(g['distinct_demo_successes'] >= 2 for g in completed)/len(completed) if completed else None,
        'distinct_successful_members': success, 'observed_attempted_slots': observed,
        'attempted_instances': observed if complete_denominator else None,
        'attempt_denominator_complete': complete_denominator,
        'successful_members_per_attempted_instance': success/observed if observed and complete_denominator else None,
        'observed_native_draws': sum(g['observed_native_draws'] for g in completed),
        'accepted': {s: accepted[s] for s in MODES}, 'standard_outcomes': dict(outcomes),
        'main_rejections': dict(reasons.most_common(15)),
        'gpu': {'terminal_collection_jobs': len(terminal_jobs), 'exact_slurm_jobs': len(exact_jobs),
            'exact_slurm_gpu_hours': hours, 'accepted_from_exact_slurm_jobs': covered_accepted,
            'slurm_accepted_per_gpu_hour': covered_accepted/hours if hours else None,
            'all_backend_collection_gpu_hours': hours if all_hours_known else None,
            'all_backend_accepted_per_gpu_hour': all_accepted/hours if hours and all_hours_known and all_groups_covered else None,
            'missing_job_ids': [j['job_id'] for j in terminal_jobs if j not in exact_jobs],
            'setup_failed_jobs_with_no_activated_group': sum(j['terminal'] and not j['activated_groups'] for j in cohort_jobs),
            'scope': 'Collection allocations including startup, failed draws and demos. Shared replay/bootstrap/rental boot and idle are not allocated to groups; end-to-end rate unavailable.',
            'end_to_end_accepted_per_gpu_hour': None}}


def revision_comparisons(cohorts):
    classes=defaultdict(list)
    for r in cohorts:
        if r['completed_groups'] and r['first_attempt_at']:
            classes[(r['class_id'],r['split'])].append(r)
    result=[]
    for (class_id,split),rows in classes.items():
        rows.sort(key=lambda r:r['first_attempt_at'])
        for before,after in zip(rows,rows[1:]):
            result.append({'class_id':class_id,'task':after['task'],'split':split,
                'before_revision':before['runtime_revision'],'after_revision':after['runtime_revision'],
                'before_completed':before['completed_groups'],'after_completed':after['completed_groups'],
                'before_at_least_two_percent':before['at_least_two_percent'],'after_at_least_two_percent':after['at_least_two_percent'],
                'change_percentage_points':after['at_least_two_percent']-before['at_least_two_percent'],
                'note':'Observational comparison; goal difficulty and cohort sizes differ. Not a controlled estimate of a fix.'})
    return result


def snapshot(root):
    root = Path(root).resolve(); started = time.time(); ev = Evidence(root)
    accepted, accepted_groups, allocation_sources = snapshot_allocations(root)
    groups = {}; jobs = []; excluded_camera = sum(str(r.get('task','')).startswith('camera_') for r in accepted)
    for path in sorted((root/'groups').glob('*/goal_plan.json')):
        try:
            plan = ev.read(path, 'goal_plan'); cfg = ev.read(path.with_name('campaign_config.json'), 'group_config')
            require(plan['campaign_id'] == cfg['campaign_id'] == CAMPAIGN, 'wrong campaign')
            if cfg['template']['task'].startswith('camera_'): continue
            gid = plan['group_id']; require(gid == path.parent.name, 'goal group directory identity differs')
            groups[gid] = fresh_group(gid, cfg, plan, path)
        except (OSError, ValueError, KeyError, TypeError) as error: ev.error(path, error)
    accounting_path = root/'accounting/job_gpu_time.json'
    accounting = ev.read(accounting_path, 'gpu_accounting') if accounting_path.exists() else {}
    account = {r['job_id']: r for r in accounting.get('records', [])}
    validation_path = root/'validation/controller.json'
    validation = ev.read(validation_path, 'validation_controller') if validation_path.exists() else {}
    gates_by_key = {}; gate_states = defaultdict(list)
    for jid, rec in validation.get('gates', {}).items():
        gate_states[jid].append({'status': rec.get('status'), 'reason': rec.get('error') or rec.get('reason'),
                                'retry': bool(rec.get('attempt_history') or rec.get('retry_request'))})
        gate = rec.get('gate') or {}
        if gate.get('success') is not True or gate.get('replay_mode') != 'physics_action_open_loop': continue
        try:
            block = ev.read(gate['instance'], 'gate_source_tro')['ispatialgym'].get('campaign')
            if not block: continue  # Ungrouped baseline gates are outside this metric.
            gates_by_key[digest(block['validation_key'])] = {'job_id': jid, 'path': rec.get('output'), 'validated_at': gate.get('validated_at')}
        except (OSError, ValueError, KeyError, TypeError) as error: ev.error(validation_path, error)
    rows = {}
    for row in allocations(root):
        jid = row['jobid']
        if jid in rows and rows[jid]['job_dir'] != row['job_dir']:
            ev.error(row['ledger'], ValueError('conflicting allocation job directory')); continue
        rows[jid] = row
        if row.get('ledger') and row['ledger'] not in ev.sources:
            ev.bytes(row['ledger'], 'allocation_ledger')
    unresolved = []
    for jid, row in rows.items():
        job = Path(row['job_dir']).resolve()
        try:
            require(job.is_relative_to(root/'out'), 'job outside campaign output')
            env_path = job/'job.env'
            if not env_path.exists():
                unresolved.append({'job_id': jid, 'reason': 'job.env unavailable; group scope not inferred from requested count'}); continue
            ev.bytes(env_path, 'job_environment_identity')  # Hash only; never expose environment contents.
            env = read_env(env_path)
            queue_path = job/'campaign_inputs/queue.json'
            queue_value = env.get('ISPATIALGYM_CAMPAIGN_GOAL_QUEUE')
            plan_value = env.get('ISPATIALGYM_CAMPAIGN_GOAL_PLAN')
            if not queue_value and not plan_value: continue
            queue = None; events = []; gids = []
            if queue_value:
                if not queue_path.exists(): queue_path = Path(queue_value)
                queue = ev.read(queue_path, 'executed_queue')
                require(queue['campaign_id'] == CAMPAIGN, 'queue campaign differs')
                qconfig = ev.read(queue['config_path'], 'queue_config')
                require(ev.sources[str(Path(queue['config_path']).resolve())]['sha256'] == queue['config_sha256'], 'queue config hash mismatch')
                gids = []
                for entry in queue['entries']:
                    plan = ev.read(entry['path'], 'queued_plan')
                    require(ev.sources[str(Path(entry['path']).resolve())]['sha256'] == entry['sha256'], 'queued plan hash mismatch')
                    require(plan['campaign_id'] == CAMPAIGN and plan['split'] == qconfig['split'], 'queued plan scope differs')
                    gids.append(plan['group_id'])
                ep = queue_path.with_suffix('.events.jsonl')
                if ep.exists():
                    events = ev.lines(ep, 'queue_events')
                    for event in events:
                        require(event['queue_sha256'] == ev.sources[str(queue_path.resolve())]['sha256'], 'queue event hash mismatch')
                        require(event['group_id'] in gids, 'event group outside executed queue')
            else:
                plan = ev.read(plan_value, 'executed_plan'); gids = [plan['group_id']]
            if not gids or any(g not in groups for g in gids):
                unresolved.append({'job_id': jid, 'reason': 'executed group absent from authored registry', 'groups': gids}); continue
            a = account.get(jid, {}); state = a.get('state', '')
            done_file = job/'job_stats.json'; stats = ev.read(done_file, 'job_statistics') if done_file.exists() else {}
            terminal = state.split()[0] in TERMINAL if state else False
            terminal = terminal or stats.get('job_wall_s') is not None
            if not terminal and (job/'sampling.done').exists():
                terminal = all((job/f'collect_c{i}.rc').exists() for i in range(1, int(row.get('kcollect') or 1)+1))
            j = {'job_id': jid, 'group_ids': gids, 'job_dir': str(job), 'runtime_revision': Path(row['code']).name.removeprefix('trunk_'),
                'backend': a.get('backend', 'vast' if jid.startswith('900000') else 'slurm'), 'state': state or 'unavailable',
                'terminal': terminal, 'allocated_gpu_seconds': a.get('allocated_gpu_seconds'),
                'missing_time_components': a.get('missing_time_components', []), 'activated_groups': [],
                'unassigned_failure_rows': 0, 'sampling_rejections': Counter(), 'validation': gate_states.get(jid, [])}
            jobs.append(j)
            closed = {}; active = set()
            for event in events:
                gid = event['group_id']
                if event['event'] == 'activated':
                    active.add(gid); g = groups[gid]; g['first_attempt_at'] = min(g['first_attempt_at'] or event['created_at'], event['created_at'])
                if event['event'] == 'closed':
                    require(gid not in closed, 'duplicate queue closure'); closed[gid] = event
            for gid in gids:
                g = groups[gid]; g['job_ids'].add(jid); g['runtime_revision'] = j['runtime_revision']
                g['pending_execution'] |= not terminal and state in ('PENDING', 'queued', 'staging')
                g['running_execution'] |= not terminal and state not in ('PENDING', 'queued', 'staging')
            # Structured reject rows name actual attempted slots, never requested capacity.
            for path in sorted((job/'inst').rglob('campaign_sampling_failures.jsonl')):
                for rec in ev.lines(path, 'sampling_rejections'):
                    reason = str(rec.get('reason', 'unspecified'))
                    if reason in CONTROL: continue
                    j['sampling_rejections'][reason] += 1
                    gid = rec.get('goal_group_id')
                    if gid is None and not queue: gid = gids[0]
                    if gid not in gids:
                        j['unassigned_failure_rows'] += 1; continue
                    g = groups[gid]; active.add(gid); g['rejections'][reason] += 1
                    g['attempt_slots'].add((jid, rec['index']))
                    g['native_attempts'].add((jid, rec['index'], rec['attempt'], rec['seed']))
                    at = rec.get('created_at'); g['first_attempt_at'] = min(g['first_attempt_at'] or at, at) if at else g['first_attempt_at']
                    g['last_attempt_at'] = max(g['last_attempt_at'] or 0, at or 0)
            for path in sorted((job/'inst').rglob('campaign_attempts.jsonl')):
                for rec in ev.lines(path, 'native_attempt_events'):
                    gid = rec.get('goal_group_id')
                    if gid not in gids or rec.get('event') != 'started': continue
                    g = groups[gid]; active.add(gid); g['exact_instrumentation_jobs'].add(jid)
                    require(rec['validation_key'] == g['validation_key'], 'attempt event validation key differs')
                    g['exact_slots'].add((jid, rec['index']))
                    g['attempt_slots'].add((jid, rec['index'])); g['exact_started'].add(rec['attempt_id'])
                    g['native_attempts'].add((jid, rec['index'], rec['attempt'], rec['seed']))
            for path in sorted((job/'inst').rglob('*-tro_state.json')):
                raw = ev.read(path, 'member_tro')['ispatialgym']; block = raw.get('campaign') or {}; gid = block.get('goal_group_id')
                if gid not in gids: continue
                g = groups[gid]; active.add(gid); iid = raw['instance_id']
                at=block.get('created_at')
                if isinstance(at,(int,float)):
                    g['first_attempt_at']=min(g['first_attempt_at'] or at,at)
                    g['last_attempt_at']=max(g['last_attempt_at'] or 0,at)
                require(block['validation_key'] == g['validation_key'], 'member validation key differs from group')
                index = int(iid.rsplit('__',1)[-1]); g['attempt_slots'].add((jid,index))
                report_path = job/'demo'/iid/'meta.json'; report = ev.read(report_path, 'demo_report') if report_path.exists() else {}
                success = False; start = None; failure = None
                if report.get('success') is True:
                    try:
                        require(report['instance_id'] == iid, 'demo identity mismatch')
                        require(Path(report['provenance']['isg_root']).resolve() == Path(row['code']).resolve(), 'demo used another runtime snapshot')
                        report_passed(report, 'oracle'); start = member_start(block); success = True
                    except (ValueError, KeyError, TypeError) as error: failure = 'demo_success_rejected: '+str(error)
                elif report.get('success') is False:
                    failure = str(report.get('reason') or report.get('error') or report.get('status') or 'demo_failed')
                if failure: g['demo_rejections'][failure] += 1
                g['members'][iid] = {'instance_id': iid, 'job_id': jid, 'demo_success': success, 'start_fingerprint': start,
                    'demo_finished': isinstance(report.get('success'), bool), 'report': str(report_path), 'tro': str(path), 'failure': failure}
            for gid in gids:
                g = groups[gid]
                if gid in active:
                    g['activated'] = True; g['executed_jobs'].add(jid)
                    if gid in closed:
                        observed_members = sum(m['job_id'] == jid for m in g['members'].values())
                        if closed[gid]['written'] != observed_members:
                            g['evidence_incomplete'].append('published TRO count differs from durable closure in job '+jid)
                        g['sampling_closed'] = True; g['closure_sources'].add(str(queue_path.with_suffix('.events.jsonl')))
                        g['last_attempt_at'] = max(g['last_attempt_at'] or 0, closed[gid]['created_at'])
                    if terminal:
                        g['sampling_closed'] = True; g['terminal'] = True
                        g['closure_sources'].add('terminal_job:'+jid)
            j['activated_groups'] = sorted(active)
            j['sampling_rejections'] = dict(j['sampling_rejections'])
        except (OSError, ValueError, KeyError, TypeError) as error: ev.error(job, error)
    for row in accepted:
        if str(row.get('task', '')).startswith('camera_'): continue
        gid = row.get('group_id')
        if gid not in groups:
            ev.error(root/'dataset/allocations.sqlite', ValueError('accepted group not observed: '+str(gid))); continue
        g = groups[gid]; g['accepted'][row['split']] += 1; g['accepted_ids'].append(row['instance_id'])
    for path in sorted((root/'controllers').glob('*/state.json')):
        try:controller_outcomes(ev.read(path,'controller_group_outcomes'),groups,str(path))
        except (OSError,ValueError,KeyError,TypeError) as error:ev.error(path,error)
    for gid,g in groups.items():
        for name in ('release_block.json', 'claim_error.json'):
            path = root/'groups'/gid/name
            if path.exists():
                rec = ev.read(path, 'acceptance_block'); g['acceptance_reasons'].append(str(rec.get('reason') or rec.get('error') or rec.get('message') or name))
        g['distinct_demo_successes'] = len({m['start_fingerprint'] for m in g['members'].values() if m['demo_success']})
        g['raw_demo_successes'] = sum(m['demo_success'] for m in g['members'].values())
        g['duplicate_successful_members'] = g['raw_demo_successes'] - g['distinct_demo_successes']
        finished = all(m['demo_finished'] for m in g['members'].values())
        accepted_observed = all(iid in g['members'] and g['members'][iid]['demo_success'] for iid in g['accepted_ids'])
        if g['evidence_incomplete']:
            g['status'] = 'unavailable_member_evidence'
        elif g['accepted'] and not accepted_observed:
            g['status'] = 'unavailable_member_evidence'
            ev.error(g['plan'], ValueError('committed members missing from observed successful demo evidence'))
        elif g['accepted']:
            g['status'] = 'completed'; g['activated'] = True
        elif g['activated'] and g['sampling_closed'] and (finished or g['terminal']): g['status'] = 'completed'
        elif g['activated']: g['status'] = 'running_or_collecting'
        elif g['running_execution']: g['status'] = 'running_job_unstarted_group'
        elif g['pending_execution']: g['status'] = 'pending'
        elif g['job_ids']: g['status'] = 'unstarted_after_job_end'
        else: g['status'] = 'authored_unsubmitted'
        g['attempted'] = g['activated']; g['observed_attempted_slots'] = len(g.pop('attempt_slots'))
        g['observed_native_draws'] = len(g.pop('native_attempts')) + len(g['members']) if not g['exact_started'] else len(g['exact_started'])
        g['attempt_denominator_complete'] = bool(g['executed_jobs']) and g['executed_jobs'] <= g['exact_instrumentation_jobs']
        if g['attempt_denominator_complete']:g['observed_attempted_slots']=len(g['exact_slots'])
        g['physics_gate'] = gates_by_key.get(digest(g['validation_key']))
        if g['accepted'].get('eval_standard') and g['accepted'].get('train'): outcome = 'eval_with_training_siblings'
        elif g['split'] == 'eval_standard' and g['accepted'].get('train') == 1 and not g['accepted'].get('eval_standard'): outcome = 'training_only_singleton'
        elif g['accepted']: outcome = 'accepted_other'
        elif g['status'] != 'completed': outcome = 'not_completed'
        elif not g['distinct_demo_successes']: outcome = 'no_distinct_successful_demo'
        elif not g['physics_gate']: outcome = 'blocked_replay'
        else: outcome = 'blocked_acceptance_or_quota'
        g['acceptance_outcome'] = outcome
        if outcome == 'blocked_replay': g['acceptance_reasons'].append('exact_template_physics_gate_missing_or_failed')
        if outcome == 'blocked_acceptance_or_quota': g['acceptance_reasons'].append('no_committed_allocation_despite_demo_and_replay')
        g['published_members'] = len(g['members']); g['members'] = list(g['members'].values())
        for k in ('job_ids','closure_sources','exact_instrumentation_jobs','executed_jobs','exact_started','exact_slots'): g[k] = sorted(g[k])
        for k in ('accepted','rejections','demo_rejections'): g[k] = dict(g[k])
        g['acceptance_reasons']=sorted(set(g['acceptance_reasons']))
    values = list(groups.values()); cohorts = defaultdict(list)
    for g in values: cohorts[(g['mode'],g['split'],g['task'],g['class_id'],g['runtime_revision'])].append(g)
    cohort_rows = [dict(zip(('mode','split','task','class_id','runtime_revision'),k), **summarize(v,jobs),
        first_attempt_at=min((g['first_attempt_at'] for g in v if g['first_attempt_at']),default=None),
        last_attempt_at=max((g['last_attempt_at'] for g in v if g['last_attempt_at']),default=None)) for k,v in sorted(cohorts.items())]
    tasks=defaultdict(list)
    for g in values:tasks[(g['mode'],g['split'],g['task'],g['runtime_revision'])].append(g)
    task_rows=[dict(zip(('mode','split','task','runtime_revision'),k),**summarize(v,jobs)) for k,v in sorted(tasks.items())]
    return {'schema':'isg-goal-group-yield-v1','campaign_id':CAMPAIGN,
        'observed_at':datetime.now(timezone.utc).isoformat(),'duration_seconds':time.time()-started,
        'definitions':{'completed':'Sampling membership closed and all demo reports terminal, or collection job terminal; acceptance remains a separate gate.',
            'attempted':'At least one durable queue activation, native attempt/rejection or published member. Authored/unactivated groups excluded.',
            'successful':'Oracle report passes success + hard QA + restoration checks, deduplicated by requested goal/start and perturbations using acceptance fingerprints.',
            'attempt_denominator':'Historical failure/TRO union is an observed LOWER BOUND of group-slot attempts. Requested n is never used. One index may straddle groups. Exact ratios require started-attempt journals.',
            'gpu':'Exact Slurm collection allocation time includes startup and failures. Vast per-group startup/retry attribution and shared replay/bootstrap costs are incomplete; all-backend/end-to-end rates unavailable.',
            'revision':'Executed immutable runtime snapshot, not the current checkout. Comparing revisions is observational, not a controlled effect estimate.',
            'exclusion':'Camera tasks and ungrouped baseline/template validation jobs are excluded.'},
        'summary':{m:summarize([g for g in values if g['mode']==m],jobs) for m in ('standard_eval','train_only','held_out')},
        'cohorts':cohort_rows,'tasks':task_rows,'revision_comparisons':revision_comparisons(cohort_rows),
        'groups':values,'jobs':jobs,'unresolved_jobs':unresolved,
        'reconciliation':{'noncamera_committed_instances':sum(not str(r.get('task','')).startswith('camera_') for r in accepted),
            'group_accepted_instances':sum(sum(g['accepted'].values()) for g in values),'excluded_camera_accepted_instances':excluded_camera,
            'groups_with_unavailable_evidence':sum(g['status']=='unavailable_member_evidence' for g in values),
            'unassigned_sampling_rejections':sum(j['unassigned_failure_rows'] for j in jobs)},
        'gpu_accounting_observed_at':accounting.get('observed_at_unix'),
        'allocation_sources':allocation_sources,'sources':list(ev.sources.values()),'errors':ev.errors}


def report_text(value):
    from zoneinfo import ZoneInfo
    stamp=datetime.fromisoformat(value['observed_at']).astimezone(ZoneInfo('America/Chicago')).strftime('%b %d, %Y %I:%M:%S %p %Z')
    lines=['# Goal-group yield — campaign 20261110','',f'Observed {stamp}. Camera tasks excluded.',
           '', '| Group type | Completed | 0 / 1 / 2 / 3 successes | ≥2 successes | Accepted eval | Accepted train |',
           '|---|---:|---|---:|---:|---:|']
    for mode,r in value['summary'].items():
        a=r['accepted'];h=' / '.join(str(r['success_histogram'][str(i)]) for i in range(4));pc=r['at_least_two_percent']
        lines.append(f"| {mode} | {r['completed_groups']} | {h} | {pc:.1f}% | {a['eval_standard']+a['eval_model']+a['eval_category']} | {a['train']} |" if pc is not None else f'| {mode} | 0 | {h} | unavailable | 0 | 0 |')
    for mode,r in value['summary'].items():
        g=r['gpu'];rate=g['slurm_accepted_per_gpu_hour']
        lines += ['',f'## {mode}', '',f"Attempted groups: {r['groups_attempted']}. States: {json.dumps(r['pending_running_unstarted'])}.",
            f"Distinct successful members: {r['distinct_successful_members']}; observed attempted group-slots: ≥{r['observed_attempted_slots']}. "
            + ('Exact started-attempt denominator available.' if r['attempt_denominator_complete'] else 'Exact historical denominator unavailable; no ratio inferred from requested counts.'),
            f"Committed outcomes: {json.dumps(r['standard_outcomes'])}.",
            f"Slurm collection subset: {g['exact_slurm_gpu_hours']:.3f} GPU-hours including startup and failures; {g['accepted_from_exact_slurm_jobs']} accepted members; "
            +(f'{rate:.3f} accepted/GPU-hour.' if rate is not None else 'rate unavailable.'),
            'All-backend/end-to-end rate unavailable: historical Vast startup/retry and shared replay/bootstrap allocation incomplete.',
            'Main recorded rejection reasons: '+json.dumps(r['main_rejections'])]
    lines += ['', '## Definitions and evidence', '']+[f'- **{k}**: {v}' for k,v in value['definitions'].items()]
    lines += ['', '## Runtime comparisons', '', 'Observed differences; difficulty and sample sizes vary. These are not causal estimates of fixes.']
    for r in value['revision_comparisons']:
        lines.append(f"- {r['class_id']} ({r['split']}): {r['before_revision']} {r['before_at_least_two_percent']:.1f}% ({r['before_completed']} groups) → "
                     f"{r['after_revision']} {r['after_at_least_two_percent']:.1f}% ({r['after_completed']} groups), {r['change_percentage_points']:+.1f} percentage points.")
    lines += ['',f"Reconciliation: {json.dumps(value['reconciliation'])}.",
        f"Evidence errors: {len(value['errors'])}; unresolved allocation scopes: {len(value['unresolved_jobs'])}.",
        'Class/runtime cohorts, individual groups, job costs, retry observations, and source hashes are in goal_group_yield.json.',
        'Live browser: http://127.0.0.1:8765/#goal-yield. Existing LIVE.md and live.json are preserved.']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--interval',type=int,default=0)
    args=p.parse_args(); destination=args.root/'infra/goal_group_yield.json'
    while True:
        try:
            value=snapshot(args.root);write_json_atomic(destination,value)
            report=destination.with_name('GOAL_GROUP_YIELD.md');temp=report.with_suffix('.tmp')
            temp.write_text(report_text(value));temp.replace(report)
            print(json.dumps({k:value[k] for k in ('observed_at','duration_seconds','reconciliation','summary')}),flush=True)
        except Exception as error:
            write_json_atomic(destination.with_name('goal_group_yield_error.json'),{'observed_at':datetime.now(timezone.utc).isoformat(),'error':type(error).__name__+': '+str(error)})
            if not args.interval:raise
        if not args.interval:return
        time.sleep(max(30,args.interval))


if __name__=='__main__':main()
