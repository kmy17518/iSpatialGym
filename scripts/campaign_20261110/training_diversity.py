"""Exact index tickets that preserve planned training diversity capacity.

Tickets reserve quota only. They never authorize production or replace native,
replay, applicable-path or shared-pipeline evidence. Unused reservations remain
charged until a separately verified terminal reconciliation exists.
"""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
from scripts.campaign_20261110.contracts import digest, file_digest, require, task_for_alias, task_quotas, write_json_atomic
from scripts.campaign_20261110.storage import atomic_directory_lock

CAMPAIGN = 'campaign_20261110'
SCHEMA = 'isg-training-diversity-plan-v1'
DIMENSIONS = ('class', 'location', 'model', 'scale')


def _sha(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'invalid exact SHA-256')


def _owned(root, path):
    p = Path(path).resolve()
    require(p.is_relative_to(Path(root).resolve()), 'diversity input outside campaign')
    return p


def _positive(value, name):
    require(type(value) is int and value > 0, name + ' must be a positive integer')
    return value


def cell_dimensions(cell):
    for key in ('id', 'class_id', 'scene', 'room', 'support_id', 'target_scale_id', 'support_height_id'):
        require(isinstance(cell.get(key), str) and bool(cell[key]), 'missing diversity cell ' + key)
    require(isinstance(cell.get('model_keys'), list) and all(isinstance(x, str) and x for x in cell['model_keys']),
            'invalid model identities')
    require(len(cell['model_keys']) == len(set(cell['model_keys'])), 'duplicate model identity')
    return {'class': [cell['class_id']],
            'location': [cell['scene'] + '/' + cell['room'] + '/' + cell['support_id']],
            'model': cell['model_keys'],
            'scale': [cell['target_scale_id'] + '/' + cell['support_height_id']]}


def load_plan(root, path, sha256):
    _sha(sha256); path = _owned(root, path)
    require(file_digest(path) == sha256, 'diversity plan changed')
    plan = json.loads(path.read_text())
    require(plan.get('schema') == SCHEMA and plan.get('campaign_id') == CAMPAIGN, 'invalid diversity plan identity')
    quota = next((q for q in task_quotas() if q.task == plan.get('task')), None)
    require(quota is not None, 'unknown diversity task')
    budget = _positive(plan.get('task_budget'), 'task budget')
    require(budget <= quota.train_instances, 'task budget exceeds owner quota')
    classes = plan.get('class_budgets')
    require(isinstance(classes, dict) and classes, 'explicit class budgets required')
    for cls, cap in classes.items():
        require(isinstance(cls, str) and bool(cls) and task_for_alias(cls) == plan['task'], 'invalid class identity'); _positive(cap, 'class budget')
        require(cap <= quota.train_instances // quota.standard_classes, 'class budget exceeds original per-class allocation')
    require(sum(classes.values()) <= budget, 'class budgets exceed task budget')
    inventory = plan.get('dimension_inventory')
    require(plan.get('inventory_complete') is True and isinstance(inventory, dict)
            and set(inventory) == set(DIMENSIONS), 'complete dimension inventory required')
    for key, values in inventory.items():
        require(isinstance(values, list) and all(isinstance(x, str) and x for x in values)
                and len(values) == len(set(values)), 'invalid dimension inventory')
        require(values or key == 'model' and plan['task'] == 'camera_relocalization-standard', 'empty required dimension')
    require(set(inventory['class']) == set(classes), 'class inventory and budgets differ')
    cells = plan.get('cells'); future = plan.get('reserved_future_capacity')
    require(isinstance(cells, list) and cells and isinstance(future, list), 'explicit cells and future capacity required')
    covered = {k: set() for k in DIMENSIONS}; per_class = Counter(); ids = set(); total = 0
    for is_future, rows in ((False, cells), (True, future)):
        for row in rows:
            require(isinstance(row, dict) and isinstance(row.get('id'), str) and bool(row['id']) and row['id'] not in ids, 'duplicate capacity pool')
            ids.add(row['id']); cls = row.get('class_id'); require(cls in classes, 'capacity class outside budgets')
            amount = _positive(row.get('capacity'), 'cell/future capacity'); total += amount; per_class[cls] += amount
            dims = row.get('dimensions') if is_future else cell_dimensions(row)
            require(isinstance(dims, dict) and set(dims) == set(DIMENSIONS), 'future pool needs explicit dimension IDs')
            require(dims['class'] == [cls], 'future pool class differs')
            for key in DIMENSIONS:
                require(isinstance(dims[key], list) and all(isinstance(x,str) and x in inventory[key] for x in dims[key]),
                        'capacity references unknown dimension')
                covered[key].update(dims[key])
    require(total <= budget and all(per_class[c] <= classes[c] for c in classes), 'capacity plan exceeds task/class budget')
    require(all(covered[k] == set(inventory[k]) for k in DIMENSIONS), 'unmet dimension lacks reserved capacity')
    return plan


def _records(root, task):
    from scripts.campaign_20261110.training_eligibility import eligible_training_records
    source = Path(root)/'dataset/allocations.sqlite'
    require(source.is_file(), 'authoritative eligible acceptance unavailable')
    with tempfile.TemporaryDirectory(prefix='isg-diversity-', dir='/tmp') as directory:
        local = Path(directory)/'state.sqlite'; shutil.copyfile(source, local)
        with sqlite3.connect(f'file:{local}?mode=ro', uri=True) as db:
            return eligible_training_records(db, task=task, root=root)


def _read_state(root):
    path = Path(root)/'controllers/training_diversity/state.json'
    if not path.exists(): return {'campaign_id': CAMPAIGN, 'plans': {}, 'tickets': {}}
    value = json.loads(path.read_text()); require(value.get('campaign_id') == CAMPAIGN, 'foreign ticket ledger')
    return value


def _ticket_rows(root, state):
    rows = []
    for request, ref in state['tickets'].items():
        path = _owned(root, ref['path']); require(file_digest(path) == ref['sha256'], 'issued ticket changed')
        row = json.loads(path.read_text()); require(row['request_id'] == request and row['campaign_id'] == CAMPAIGN, 'ticket identity differs')
        retirement = state.get('retirements', {}).get(request)
        if retirement:
            receipt = _read_reference(root, retirement)
            require(receipt.get('schema') == 'isg-training-ticket-retirement-v1' and receipt.get('campaign_id') == CAMPAIGN
                    and receipt.get('ticket') == ref, 'ticket retirement identity differs')
            released = receipt['released_indices']
            require(len(released) == len(set(released)) and all(type(i) is int and
                    row['start'] <= i < row['start']+row['count'] for i in released),
                    'retirement releases an invalid position')
            row = {**row, 'released_indices': released}
        rows.append(row)
    return rows


def _index(record):
    value = record.get('index')
    if type(value) is int: return value
    match = re.search(r'__(\d+)$', record.get('instance_id', ''))
    return int(match[1]) if match else None


def _budget(root, plan, rows):
    accepted = _records(root, plan['task']); counts = Counter(r['class_id'] for r in accepted)
    indices = set()
    for record in accepted:
        value = _index(record)
        if value is not None: indices.add(value)
    pending = Counter()
    for row in rows:
        if row['task'] == plan['task']:
            released = set(row.get('released_indices', []))
            require(not released & indices, 'retired unused position was accepted later')
            pending[row['class_id']] += sum(i not in indices and i not in released
                for i in range(row['start'], row['start']+row['count']))
    reserved = Counter()
    for pool in plan['reserved_future_capacity']: reserved[pool['class_id']] += pool['capacity']
    require(len(accepted)+sum(pending.values())+sum(reserved.values()) <= plan['task_budget'],
            'eligible accepted plus pending tickets would consume future task capacity')
    require(all(counts[c]+pending[c]+reserved[c] <= cap for c,cap in plan['class_budgets'].items()),
            'eligible accepted plus pending tickets would consume future class capacity')


def reserve_ticket(root, request_id, config_sha256, start, count, cell_id, plan_path, plan_sha256):
    root = Path(root).resolve(); _sha(config_sha256)
    require(isinstance(request_id, str) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*', request_id), 'invalid ticket request ID')
    _positive(start, 'start index'); _positive(count, 'instance count')
    plan_path = _owned(root, plan_path); plan = load_plan(root, plan_path, plan_sha256)
    cell = next((x for x in plan['cells'] if x['id'] == cell_id), None); require(cell is not None, 'cell not planned')
    # The common index allocator remains authoritative; this is a quota ticket,
    # not a new index allocator or a permit to recycle historic identifiers.
    reservations = json.loads((root/'manifests/family_index_reservations.json').read_text())['reservations']
    require(any(r['start'] <= start and start+count <= r['start']+r['n'] for r in reservations.values()),
            'ticket range lacks an existing common index reservation')
    directory = root/'controllers/training_diversity'; path = directory/'tickets'/(request_id+'.json')
    identity = {'campaign_id': CAMPAIGN, 'request_id': request_id, 'config_sha256': config_sha256,
                'start': start, 'count': count, 'cell_id': cell_id, 'task': plan['task'], 'class_id': cell['class_id'],
                'plan': {'path': str(plan_path), 'sha256': plan_sha256}, 'dimensions': cell_dimensions(cell),
                'production_authorization': False}
    with atomic_directory_lock(directory/'state.lockdir'):
        state = _read_state(root); rows = _ticket_rows(root, state)
        if request_id in state['tickets']:
            old = next(x for x in rows if x['request_id'] == request_id)
            require(all(old.get(k) == v for k,v in identity.items()), 'idempotent ticket request changed')
            require(request_id not in state.get('retirements', {}), 'retired ticket cannot be reissued')
            return state['tickets'][request_id]
        prior = state['plans'].get(plan['task'])
        require(prior is None or prior == identity['plan'], 'task plan revision requires explicit ticket reconciliation')
        require(not any(start < r['start']+r['count'] and r['start'] < start+count for r in rows), 'ticket index range overlaps')
        issued = sum(r['count']-len(r.get('released_indices', [])) for r in rows
                     if r['task'] == plan['task'] and r['cell_id'] == cell_id)
        require(issued+count <= cell['capacity'], 'cell capacity exhausted')
        _budget(root, plan, rows+[identity])
        if path.exists():
            orphan = json.loads(path.read_text())
            require(all(orphan.get(k) == v for k,v in identity.items()), 'unregistered ticket bytes differ')
        else: write_json_atomic(path, {**identity, 'reserved_at': time.time()})
        ref = {'path': str(path), 'sha256': file_digest(path)}
        state['plans'][plan['task']] = identity['plan']; state['tickets'][request_id] = ref
        write_json_atomic(directory/'state.json', state)
        return ref


def check_ticket_identity(root, config_sha256, start, count, ticket_ref):
    """Immutable membership only, for already-committed acceptance audits.

    Dispatch must call check_ticket, which additionally rechecks live budgets.
    This nonrecursive identity check cannot reserve or authorize work.
    """
    root = Path(root).resolve(); _sha(config_sha256); _positive(start, 'start index'); _positive(count, 'instance count')
    path = _owned(root, ticket_ref['path']); _sha(ticket_ref['sha256'])
    require(file_digest(path) == ticket_ref['sha256'], 'ticket bytes changed')
    row = json.loads(path.read_text()); state = _read_state(root)
    require(state['tickets'].get(row.get('request_id')) == ticket_ref, 'ticket is not committed')
    require(row['campaign_id'] == CAMPAIGN and row['config_sha256'] == config_sha256
            and (row['start'], row['count']) == (start, count), 'ticket exact configuration/range differs')
    plan = load_plan(root, row['plan']['path'], row['plan']['sha256'])
    require(state['plans'].get(plan['task']) == row['plan'], 'ticket plan is no longer active')
    return {'ticket': ticket_ref, 'task': row['task'], 'class_id': row['class_id'],
            'cell_id': row['cell_id'], 'start': start, 'count': count,
            'config_sha256': config_sha256, 'production_authorization': False}


def check_ticket(root, config_sha256, start, count, ticket_ref):
    """Read-only exact-range and live-budget check for dispatch."""
    proof = check_ticket_identity(root, config_sha256, start, count, ticket_ref)
    state = _read_state(root)
    ticket = _read_reference(root, ticket_ref)
    require(ticket['request_id'] not in state.get('retirements', {}),
            'retired ticket cannot dispatch or accept new instances')
    plan_ref = state['plans'][proof['task']]
    plan = load_plan(root, plan_ref['path'], plan_ref['sha256'])
    _budget(root, plan, _ticket_rows(root, state))
    return proof


def _read_reference(root, reference):
    path = _owned(root, reference['path']); _sha(reference['sha256'])
    require(path.is_file() and file_digest(path) == reference['sha256'], 'terminal evidence missing or changed')
    return json.loads(path.read_text())


def _controller_scope(controller, ticket_ref, ticket):
    require(controller.get('schema') == 'isg-training-ticket-controller-v1' and
            controller.get('campaign_id') == CAMPAIGN and controller.get('ticket') == ticket_ref,
            'terminal controller identity differs')
    require(bool(controller.get('owner')) and all(controller.get(k) == ticket[k]
            for k in ('config_sha256', 'start', 'count')), 'terminal controller scope differs')


def bind_ticket_controller(root, ticket_ref, controller_path):
    """Bind the real future producer before dispatch; never create its state."""
    root = Path(root).resolve(); path = _owned(root, controller_path)
    controller = json.loads(path.read_text()); ticket = _read_reference(root, ticket_ref)
    _controller_scope(controller, ticket_ref, ticket)
    require(controller.get('phase') in ('prepared', 'open'), 'initial controller must be prepared or open')
    directory = root/'controllers/training_diversity'
    with atomic_directory_lock(directory/'state.lockdir'):
        state = _read_state(root); request = ticket['request_id']
        require(state['tickets'].get(request) == ticket_ref and request not in state.get('retirements', {}),
                'controller needs an active committed ticket')
        binding = {'path': str(path), 'owner': controller['owner'], 'ticket': ticket_ref}
        bindings = state.setdefault('controllers', {})
        require(request not in bindings or bindings[request] == binding, 'ticket controller ownership changed')
        bindings[request] = binding
        write_json_atomic(directory/'state.json', state)
    return binding


def _terminal_jobs(root, controller, observations, now):
    """Check fresh original backend records, not a controller's terminal label."""
    require(observations.get('schema') == 'isg-training-ticket-terminal-observations-v1'
            and observations.get('campaign_id') == CAMPAIGN, 'terminal observation identity differs')
    def fresh(value):
        require(type(value) in (int, float) and -30 <= now-value <= 120,
                'terminal backend observation stale or missing')
    fresh(observations.get('observed_at'))
    jobs = controller.get('jobs'); observed = observations.get('jobs')
    require(isinstance(jobs, list) and isinstance(observed, list) and jobs,
            'complete execution job inventory required')
    def key(row):
        require(row.get('backend') in ('cluster', 'vast') and row.get('stage') in
                ('native', 'collection', 'replay', 'rendering'), 'unknown terminal job scope')
        require(isinstance(row.get('job_id'), str) and re.fullmatch(r'[0-9]+(?:_[0-9]+)?', row['job_id']),
                'exact terminal job id required')
        return (row['backend'], row['job_id'], row['stage'])
    wanted, supplied = [key(j) for j in jobs], [key(j) for j in observed]
    require(len(wanted) == len(set(wanted)) and len(supplied) == len(set(supplied)) and
            set(wanted) == set(supplied), 'terminal observations omit or duplicate owned stage jobs')
    for job in observed:
        raw = _read_reference(root, job['raw'])
        require(raw.get('campaign_id') == CAMPAIGN and raw.get('job_id') == job['job_id'],
                'raw terminal job identity differs')
        fresh(raw.get('observed_at'))
        if job['backend'] == 'cluster':
            require(raw.get('schema') == 'isg-training-ticket-slurm-observation-v1' and
                    raw.get('squeue_returncode') == raw.get('sacct_returncode') == 0 and
                    raw.get('squeue_stdout', '').strip() == '', 'cluster job is active or scheduler unavailable')
            lines = [line.strip().split('|') for line in raw.get('sacct_stdout', '').splitlines() if line.strip()]
            require(len(lines) == 1 and len(lines[0]) == 4 and lines[0][0] == job['job_id'],
                    'exact sacct JobIDRaw|State|End|ExitCode row required')
            _, status, ended, exit_code = lines[0]
            require(status.split()[0] in {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'DEADLINE'} and
                    ended not in ('', 'Unknown', 'None', 'N/A') and re.fullmatch(r'\d+:\d+', exit_code),
                    'cluster execution is not conclusively terminal')
        else:
            require(raw.get('schema') == 'isg-training-ticket-vast-observation-v1', 'invalid Vast terminal observation')
            require(_owned(root, raw['state']['path']) == Path(root)/'vast/state/jobs'/(job['job_id']+'.json'),
                    'Vast proof must read the current authoritative job state')
            current = _read_reference(root, raw['state'])
            done = current.get('done') or {}
            require(str(current.get('id')) == job['job_id'] and current.get('status') == 'synced' and
                    type(done.get('rc')) is int and done.get('end', 0) >= done.get('start', 1) and
                    done.get('attempt_evidence_uploaded') is True and not done.get('upload_error'),
                    'Vast execution or durable output is not terminal')


def observe_ticket_jobs(root, controller_ref, output_directory):
    """Capture real read-only backend observations; this never retires a ticket."""
    root = Path(root).resolve(); controller = _read_reference(root, controller_ref)
    require(controller.get('schema') == 'isg-training-ticket-controller-v1'
            and controller.get('campaign_id') == CAMPAIGN, 'controller observation identity differs')
    jobs = controller.get('jobs', [])
    require(jobs and all(j.get('backend') in ('cluster', 'vast') and
            isinstance(j.get('job_id'), str) and re.fullmatch(r'[0-9]+(?:_[0-9]+)?', j['job_id'])
            for j in jobs), 'exact owned jobs required for observation')
    out = _owned(root, output_directory); out.mkdir(parents=True, exist_ok=False)
    cluster_ids = sorted({j['job_id'] for j in jobs if j['backend'] == 'cluster'})
    cluster = None
    if cluster_ids:
        ids = ','.join(cluster_ids)
        commands = {'squeue': ['squeue', '--array', '-h', '-j', ids, '-o', '%i|%T|%R'],
                    'sacct': ['sacct', '-X', '-n', '-P', '-j', ids, '--format=JobIDRaw,State%40,End,ExitCode']}
        cluster = {}
        for name, command in commands.items():
            try:
                result = subprocess.run(command, text=True, capture_output=True, timeout=45)
                cluster[name] = {'command': command, 'returncode': result.returncode,
                                 'stdout': result.stdout, 'stderr': result.stderr, 'observed_at': time.time()}
            except (OSError, subprocess.TimeoutExpired) as error:
                cluster[name] = {'command': command, 'returncode': None, 'stdout': '',
                                 'stderr': str(error), 'observed_at': time.time()}
        write_json_atomic(out/'cluster_queries.json', cluster)
    rows = []
    for ordinal, job in enumerate(jobs):
        identity = {'campaign_id': CAMPAIGN, 'job_id': job['job_id'], 'observed_at': time.time()}
        if job['backend'] == 'cluster':
            identity.update(schema='isg-training-ticket-slurm-observation-v1',
                observed_at=min(cluster[k]['observed_at'] for k in ('squeue', 'sacct')),
                original_queries={'path': str(out/'cluster_queries.json'), 'sha256': file_digest(out/'cluster_queries.json')})
            for name in ('squeue', 'sacct'):
                identity[name+'_returncode'] = cluster[name]['returncode']
                identity[name+'_stdout'] = '\n'.join(line for line in cluster[name]['stdout'].splitlines()
                    if line.strip().split('|', 1)[0] == job['job_id'])
        else:
            path = root/'vast/state/jobs'/(job['job_id']+'.json')
            identity.update(schema='isg-training-ticket-vast-observation-v1',
                            state={'path': str(path), 'sha256': file_digest(path)})
        path = out/(str(ordinal)+'_'+job['backend']+'_'+job['job_id']+'.json')
        write_json_atomic(path, identity)
        rows.append({**job, 'raw': {'path': str(path), 'sha256': file_digest(path)}})
    path = out/'observations.json'
    write_json_atomic(path, {'schema': 'isg-training-ticket-terminal-observations-v1',
        'campaign_id': CAMPAIGN, 'observed_at': time.time(), 'controller': controller_ref, 'jobs': rows})
    return {'path': str(path), 'sha256': file_digest(path)}


def _committed_ticket_records(root, ticket, ticket_ref):
    """Preserve every committed position, including temporarily excluded rows."""
    source = Path(root)/'dataset/allocations.sqlite'
    require(source.is_file(), 'canonical acceptance database missing')
    with tempfile.TemporaryDirectory(prefix='isg-ticket-terminal-', dir='/tmp') as directory:
        local = Path(directory)/'accepted.sqlite'; shutil.copyfile(source, local)
        with sqlite3.connect(f'file:{local}?mode=ro', uri=True) as db:
            records = [json.loads(value) for value, in db.execute('SELECT record FROM members')]
    selected = []
    for record in records:
        index = _index(record)
        if index is not None and ticket['start'] <= index < ticket['start']+ticket['count']:
            require(record.get('training_diversity_ticket') == ticket_ref and
                    record.get('training_config_sha256') == ticket['config_sha256'] and
                    record.get('task') == ticket['task'] and record.get('class_id') == ticket['class_id'],
                    'committed ticket range contains another identity')
            selected.append(record)
    require(len({_index(r) for r in selected}) == len(selected), 'committed ticket positions duplicated')
    return selected


def retire_ticket(root, ticket_ref, controller_ref, terminal_observations_ref, *, now=None):
    """Return unused capacity only after the registered producer fully closes.

    No job is stopped, no index is reused and no accepted record is changed.
    The producer must first reconcile all dispatch, replay and acceptance work.
    The shared acceptance lock closes the precheck/commit race; claim_group must
    recheck ticket activity inside that same lock before committing new members.
    """
    root = Path(root).resolve(); now = time.time() if now is None else now
    directory = root/'controllers/training_diversity'
    with atomic_directory_lock(root/'dataset/allocations.lockdir'):
        with atomic_directory_lock(directory/'state.lockdir'):
            state = _read_state(root); ticket = _read_reference(root, ticket_ref); request = ticket['request_id']
            require(state['tickets'].get(request) == ticket_ref, 'retirement ticket is not committed')
            previous = state.get('retirements', {}).get(request)
            if previous:
                receipt = _read_reference(root, previous)
                require(receipt['ticket'] == ticket_ref and receipt['controller'] == controller_ref and
                        receipt['terminal_observations'] == terminal_observations_ref, 'retirement retry evidence changed')
                return previous
            controller = _read_reference(root, controller_ref)
            _controller_scope(controller, ticket_ref, ticket)
            require(state.get('controllers', {}).get(request) == {
                'path': str(_owned(root, controller_ref['path'])), 'owner': controller['owner'], 'ticket': ticket_ref},
                'terminal controller was not the registered owner')
            require(controller.get('phase') == 'finalized' and controller.get('pending_dispatch') == [] and
                    controller.get('pending_acceptance') == [] and bool(controller.get('next_action')),
                    'dispatch or acceptance reconciliation is unfinished')
            require(type(controller.get('observed_at')) in (int, float) and
                    -30 <= now-controller['observed_at'] <= 120, 'controller closure is stale')
            artifacts = controller.get('outcome_artifacts')
            require(isinstance(artifacts, list) and artifacts, 'original validation outcomes required')
            for reference in artifacts: _read_reference(root, reference)
            observations = _read_reference(root, terminal_observations_ref)
            _terminal_jobs(root, controller, observations, now)
            committed = _committed_ticket_records(root, ticket, ticket_ref)
            expected = {r['instance_id']: digest(r) for r in committed}
            require(controller.get('committed_records') == expected,
                    'controller acceptance closure differs from canonical committed history')
            used = {_index(r) for r in committed}
            released = sorted(set(range(ticket['start'], ticket['start']+ticket['count']))-used)
            outcomes = controller.get('unused_outcomes', {})
            require(set(outcomes) == {str(i) for i in released}, 'every unused position needs an exact terminal outcome')
            for outcome in outcomes.values():
                require(outcome.get('status') in ('native_failed', 'demo_failed', 'replay_failed',
                        'acceptance_rejected', 'never_started') and outcome.get('artifacts') and
                        bool(outcome.get('next_action')), 'unused position lacks an owned terminal diagnosis')
                for reference in outcome['artifacts']: _read_reference(root, reference)
            receipt = {'schema': 'isg-training-ticket-retirement-v1', 'campaign_id': CAMPAIGN,
                'ticket': ticket_ref, 'controller': controller_ref, 'terminal_observations': terminal_observations_ref,
                'retired_at': now, 'released_indices': released, 'committed_records': committed,
                'unused_outcomes': outcomes,
                'retired_from_dispatch_and_acceptance': True, 'indices_remain_reserved': True,
                'production_authorization': False}
            identity = {k:v for k,v in receipt.items() if k != 'retired_at'}
            path = directory/'retirements'/(request+'_'+digest(identity)[:24]+'.json')
            if path.exists():
                prior = json.loads(path.read_text())
                require({k:v for k,v in prior.items() if k != 'retired_at'} == identity,
                        'orphan retirement evidence changed')
            else: write_json_atomic(path, receipt)
            reference = {'path': str(path), 'sha256': file_digest(path)}
            state.setdefault('retirements', {})[request] = reference
            write_json_atomic(directory/'state.json', state)
            return reference


def claim_capacity(root, ticket_ref, instance_ids):
    """Record already committed eligible IDs, without releasing unused slots.

    Call only after the ordinary acceptance transaction commits. This journal is
    idempotent accounting; canonical accepted records remain the quota source.
    Failed/unused positions stay reserved until retire_ticket verifies terminal
    reconciliation. A claim cannot authorize or relabel a collection instance.
    """
    root = Path(root).resolve()
    require(isinstance(instance_ids, (list, tuple)) and instance_ids
            and all(isinstance(i, str) and i for i in instance_ids)
            and len(instance_ids) == len(set(instance_ids)), 'distinct accepted instance IDs required')
    path = _owned(root, ticket_ref['path']); _sha(ticket_ref['sha256'])
    require(file_digest(path) == ticket_ref['sha256'], 'ticket bytes changed')
    row = json.loads(path.read_text())
    directory = root/'controllers/training_diversity'
    with atomic_directory_lock(directory/'state.lockdir'):
        state = _read_state(root)
        require(state['tickets'].get(row.get('request_id')) == ticket_ref, 'ticket is not committed')
        plan = load_plan(root, row['plan']['path'], row['plan']['sha256'])
        require(state['plans'].get(plan['task']) == row['plan'], 'ticket plan is no longer active')
        accepted = {r['instance_id']: r for r in _records(root, row['task'])}
        claims = state.setdefault('claims', {}).setdefault(row['request_id'], {})
        for iid in instance_ids:
            record = accepted.get(iid)
            require(record is not None, 'instance has no committed eligible acceptance')
            index = _index(record)
            require(index is not None and row['start'] <= index < row['start']+row['count']
                    and record['class_id'] == row['class_id'], 'accepted instance outside exact ticket cell/range')
            # The acceptance layer binds release and ticket proofs. Requiring its
            # exact ticket here prevents an unrelated same-index record joining.
            require(record.get('training_diversity_ticket') == ticket_ref,
                    'accepted instance lacks exact ticket proof')
            require(record.get('training_config_sha256') == row['config_sha256'],
                    'accepted instance configuration differs')
            proof = {'index': index, 'accepted_record_sha256': digest(record)}
            require(iid not in claims or claims[iid] == proof, 'accepted claim bytes changed')
            claims[iid] = proof
        rows = _ticket_rows(root, state)
        _budget(root, plan, rows)
        released = next(r for r in rows if r['request_id'] == row['request_id']).get('released_indices', [])
        write_json_atomic(directory/'state.json', state)
        return {'ticket': ticket_ref, 'instance_ids': sorted(claims),
                'accepted_count': len(claims),
                'unused_positions_still_reserved': row['count']-len(claims)-len(released),
                'production_authorization': False}
