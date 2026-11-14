"""One-shot preparation routing under the existing coordinator's claims and lock.

Only scheduler routing changes. Native scripts, arguments, inputs and ownership
remain those of an admitted consumed-preparation ticket. This is not a worker.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import argparse
import copy
import fcntl
import getpass
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

from scripts.campaign_20261110 import dispatch_cluster_first as cluster

USERS = ('minyeong', 'shiyuc')
PARTITIONS = ('svl', 'viscam')
RESERVE_TITANS = 4
DEFAULT_DEMAND = {'cpus': 8, 'memory_mb': 48 * 1024}
# Keep the existing native scheduler exclusions, including known broken hosts.
EXCLUDED = cluster.assigned_literal(Path(__file__).parent/'infra_dispatch_bin/sbatch', 'excluded')
ROUTING = {'--account', '--partition', '--gres', '--nodelist', '--exclude', '--nice'}
SHORT = {'-A': '--account', '-p': '--partition', '-w': '--nodelist', '-x': '--exclude',
         '-c': '--cpus-per-task', '-N': '--nodes', '-n': '--ntasks', '-J': '--job-name',
         '-o': '--output', '-e': '--error', '-t': '--time'}
VALUES = ROUTING | {'--cpus-per-task', '--nodes', '--ntasks', '--job-name', '--output',
                    '--error', '--time', '--mem', '--mem-per-cpu', '--export', '--chdir',
                    '--qos', '--signal', '--open-mode', '--dependency'}
FLAGS = {'--parsable', '--requeue', '--no-requeue'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def kv(text):
    return dict(re.findall(r'(?:^|\s)(\w+)=(\S*)', text))


def tres(value):
    return dict(item.split('=', 1) for item in value.split(',') if '=' in item)


def memory_mb(value, cpus=1):
    match = re.fullmatch(r'(\d+(?:\.\d+)?)([KMGT]?)([cn]?)', str(value), re.I)
    require(match is not None, 'Unknown memory unit: '+str(value))
    result = float(match[1]) * {'': 1, 'K': 1/1024, 'M': 1, 'G': 1024, 'T': 1024**2}[match[2].upper()]
    return result * (cpus if match[3].lower() == 'c' else 1)


def sbatch_parts(command):
    require(command and command[0] == 'sbatch', 'Only exact sbatch argv is supported')
    options, values, i = [], {}, 1
    while i < len(command) and command[i].startswith('-'):
        token = command[i]
        if token == '--':
            i += 1
            break
        if token in FLAGS:
            options.append((token, [token])); i += 1
            continue
        name, equal, value = token.partition('=')
        if name in SHORT:
            name = SHORT[name]
        elif not token.startswith('--') and token[:2] in SHORT and len(token) > 2:
            name, equal, value = SHORT[token[:2]], '=', token[2:]
        require(name in VALUES, 'Unsupported scheduler option: '+token)
        consumed = [token]
        if not equal:
            require(i+1 < len(command), 'Scheduler option lacks a value')
            i += 1; value = command[i]; consumed.append(value)
        require(name not in values, 'Repeated scheduler option: '+name)
        values[name] = value; options.append((name, consumed)); i += 1
    require(i < len(command) and '--parsable' in command, 'Missing pinned script or parsable acknowledgement')
    require(values.get('--nodes', '1') == '1' and values.get('--ntasks', '1') == '1', 'Only one-node one-task preparation routes')
    require(re.fullmatch(r'gpu(?::[A-Za-z0-9_]+)?:1', values.get('--gres', '')) is not None,
            'Only one-GPU preparation routes')
    cpus = int(values.get('--cpus-per-task', '0'))
    require(cpus > 0 and ('--mem' in values) != ('--mem-per-cpu' in values), 'Explicit CPU and memory request required')
    memory = values.get('--mem', values.get('--mem-per-cpu'))
    require(not str(memory).lower().endswith(('c', 'n')), 'Sbatch memory must use explicit option scope')
    mem = memory_mb(memory)
    if '--mem-per-cpu' in values:
        mem *= cpus
    require(mem > 0, 'Unbounded memory is unsupported')
    return options, command[i:], values, {'cpus': cpus, 'memory_mb': mem}


def observe(runner=subprocess.run):
    queries = {
        'jobs': ['squeue', '-r', '-u', ','.join(USERS), '-h', '-o', '%u|%P|%T|%b|%q|%R|%i|%a|%C|%m'],
        'nodes': ['scontrol', 'show', 'nodes', '-o'],
        'partitions': ['scontrol', 'show', 'partition', '-o'],
        'qos': ['sacctmgr', '-n', '-P', 'show', 'qos', 'format=Name,MaxTRESPU,MaxJobsPU,MaxSubmitPU'],
        'associations': ['sacctmgr', '-n', '-P', 'show', 'assoc', 'where', 'user='+','.join(USERS),
                         'format=User,Account,Partition,MaxTRESPU,MaxJobs,MaxSubmit'],
    }
    began = time.time()
    def run(argv):
        return runner(argv, capture_output=True, text=True, timeout=40, check=True).stdout
    with ThreadPoolExecutor(max_workers=len(queries)) as pool:
        futures = {key: pool.submit(run, argv) for key, argv in queries.items()}
        raw = {key: value.result() for key, value in futures.items()}
    jobs = [line.split('|') for line in raw['jobs'].splitlines() if line.strip()]
    require(all(len(row) == 10 for row in jobs), 'Scheduler job schema differs')
    observations = {'observed_at': began, 'finished_at': time.time(), 'jobs': jobs,
                    'nodes': [kv(line) for line in raw['nodes'].splitlines() if line.strip()],
                    'partitions': [kv(line) for line in raw['partitions'].splitlines() if line.strip()],
                    'qos': [line.split('|') for line in raw['qos'].splitlines() if line.strip()],
                    'associations': [line.split('|') for line in raw['associations'].splitlines() if line.strip()],
                    'query_sha256': {key: hashlib.sha256(value.encode()).hexdigest() for key, value in raw.items()}}
    require(observations['nodes'], 'No node observation')
    require(all(any(row[0] == user for row in observations['associations']) for user in USERS), 'Both user associations required')
    return observations


def gpu_request(value):
    matches = re.findall(r'(?:gres/)?gpu(?::([A-Za-z0-9_]+))?:(\d+)', value)
    return [(model or None, int(count)) for model, count in matches]


def limit(value):
    return int(value) if value and value not in ('UNLIMITED', 'NONE', '(null)') else 10**9


def access_and_quotas(observation, demand=None):
    demand = demand or DEFAULT_DEMAND
    result = {}
    qos = {row[0]: row for row in observation['qos']}
    for part in observation['partitions']:
        name = part.get('PartitionName')
        if name not in PARTITIONS or part.get('State') != 'UP':
            continue
        require(part.get('QoS') in qos, 'Partition QoS not observed: '+name)
        quota = qos[part['QoS']]
        require(len(quota) >= 4, 'QoS schema differs')
        for user in USERS:
            associations = [row for row in observation['associations'] if row[0] == user
                            and (not row[2] or row[2] == name)
                            and (part.get('AllowAccounts') == 'ALL' or row[1] in part.get('AllowAccounts', '').split(','))]
            if not associations:
                continue
            preferred = 'vision' if name == 'svl' else 'viscam'
            association = next((row for row in associations if row[1] == preferred), associations[0])
            require(len(association) >= 6, 'Association schema differs')
            active = [row for row in observation['jobs'] if row[0] == user and name in row[1].split(',')]
            gpu_used = sum(sum(count for _, count in gpu_request(row[3])) for row in active)
            running = sum(row[2] in ('RUNNING', 'COMPLETING') for row in active)
            gpu_limits = [limit(tres(value).get('gres/gpu', '')) for value in (quota[1], association[3])]
            headroom = min(*(value-gpu_used for value in gpu_limits), limit(quota[2])-running,
                           limit(quota[3])-len(active), limit(association[4])-running, limit(association[5])-len(active))
            constraints = [tres(value) for value in (quota[1], association[3])]
            for constraint in constraints:
                if 'cpu' in constraint:
                    headroom = min(headroom, (int(constraint['cpu'])-sum(int(row[8]) for row in active))//demand['cpus'])
                if 'mem' in constraint:
                    used_mem = sum(memory_mb(row[9], int(row[8] or '1')) for row in active)
                    headroom = min(headroom, int((memory_mb(constraint['mem'])-used_mem)//demand['memory_mb']))
            model_slots = {}
            for model in cluster.COMPATIBLE_MODELS:
                used_model = sum(n for row in active for wanted, n in gpu_request(row[3]) if wanted in (None, model))
                model_slots[model] = max(0, min(headroom, *(limit(c.get('gres/gpu:'+model, ''))-used_model for c in constraints)))
            result[(user, name)] = {'user': user, 'partition': name, 'slurm_account': association[1],
                                   'slots': max(0, headroom), 'qos': part['QoS'],
                                   'model_slots': model_slots,
                                   'gpu_used': gpu_used, 'running_jobs': running, 'submitted_jobs': len(active)}
    return result


def inventory(observation, demand, reservations=(), excluded=()):
    nodes = {}
    for row in observation['nodes']:
        name, state = row.get('NodeName'), row.get('State', '')
        if not name or name in set(EXCLUDED) | set(excluded):
            continue
        if not any(s in state for s in ('IDLE', 'MIXED', 'ALLOCATED')) or any(s in state for s in ('DRAIN', 'DOWN', 'FAIL', 'POWER', 'RESERVED', 'PLANNED', 'MAINT', 'NOT_RESPOND')):
            continue
        parts = set(row.get('Partitions', '').split(',')) & set(PARTITIONS)
        cfg, used = tres(row.get('CfgTRES', '')), tres(row.get('AllocTRES', ''))
        models = {key.split(':', 1)[1]: int(value) for key, value in cfg.items() if key.startswith('gres/gpu:')}
        unknown = max(0, int(used.get('gres/gpu', '0'))-sum(int(v) for k, v in used.items() if k.startswith('gres/gpu:')))
        gpus = {model: max(0, count-int(used.get('gres/gpu:'+model, '0'))-unknown)
                for model, count in models.items() if model in cluster.COMPATIBLE_MODELS}
        if not parts or not any(gpus.values()):
            continue
        nodes[name] = {'node': name, 'partitions': parts, 'gpus': gpus,
                       'gpu_total': max(0, int(cfg.get('gres/gpu', '0'))-int(used.get('gres/gpu', '0'))),
                       'cpus': max(0, int(row.get('CPUEfctv', row.get('CPUTot', '0')))-int(row.get('CPUAlloc', '0'))),
                       'memory_mb': max(0, int(row.get('RealMemory', '0'))-int(row.get('AllocMem', '0')))}
    def spend(node, model, cpus, mem):
        node['gpus'][model] = max(0, node['gpus'].get(model, 0)-1)
        node['gpu_total'] = max(0, node['gpu_total']-1)
        node['cpus'] = max(0, node['cpus']-cpus); node['memory_mb'] = max(0, node['memory_mb']-mem)
    # Existing pending work also consumes prospective physical resources. Typed
    # requests on unsupported models do not consume a compatible GPU slot.
    for job in observation['jobs']:
        if job[2] != 'PENDING':
            continue
        requests = gpu_request(job[3]); count = sum(n for _, n in requests)
        cpus = max(demand['cpus'], (int(job[8] or '0')+max(count, 1)-1)//max(count, 1))
        mem = max(demand['memory_mb'], memory_mb(job[9], int(job[8] or '1'))/max(count, 1)) if count else 0
        for wanted, n in requests:
            for _ in range(n):
                options = [(node, model) for node in nodes.values() for model, free in node['gpus'].items()
                           if free and node['gpu_total'] and (wanted is None or model == wanted)
                           and node['partitions'] & set(job[1].split(','))]
                if options:
                    node, model = sorted(options, key=lambda v: (v[1] != 'titanrtx', v[0]['node'], v[1]))[0]
                    spend(node, model, cpus, mem)
    for reservation in reservations:
        if reservation['node'] in nodes:
            spend(nodes[reservation['node']], reservation['gpu_model'], reservation['cpus'], reservation['memory_mb'])
    return nodes


def node_slots(node, model, demand):
    return min(node['gpus'].get(model, 0), node['gpu_total'], node['cpus']//demand['cpus'],
               int(node['memory_mb']//demand['memory_mb']))


def reserve_titans(nodes):
    """Reserve physical Titans plus the CPU/memory needed by critical controls."""
    available = sum(node_slots(node, 'titanrtx', DEFAULT_DEMAND) for node in nodes.values())
    held = min(RESERVE_TITANS, available); remaining = held
    for name in sorted(nodes):
        node = nodes[name]; take = min(remaining, node_slots(node, 'titanrtx', DEFAULT_DEMAND))
        node['gpus']['titanrtx'] = max(0, node['gpus'].get('titanrtx', 0)-take)
        node['gpu_total'] -= take; node['cpus'] -= take*DEFAULT_DEMAND['cpus']
        node['memory_mb'] -= take*DEFAULT_DEMAND['memory_mb']; remaining -= take
    if held < RESERVE_TITANS:
        for node in nodes.values():
            node['gpus']['titanrtx'] = 0
    return available, held


def build_capacity(observation, demand=None, *, now=None, reservations=()):
    demand = dict(demand or DEFAULT_DEMAND); now = time.time() if now is None else now
    require(0 <= now-observation['observed_at'] <= 60, 'Cluster observation expired')
    access = access_and_quotas(observation, demand); nodes = inventory(observation, demand, reservations)
    titan, held = reserve_titans(nodes)
    for reservation in reservations:
        key = (reservation['user'], reservation['partition'])
        if key in access:
            access[key]['slots'] = max(0, access[key]['slots']-1)
            model = reservation['gpu_model']
            access[key]['model_slots'][model] = max(0, access[key]['model_slots'][model]-1)
    # Count feasible typed assignments, sharing node resources across partitions
    # and users. This is a conservative lower bound, never a sum of overlapping
    # per-account physical capacity or unavailable typed-quota slots.
    def assignable(users):
        remaining = copy.deepcopy(nodes); quotas = copy.deepcopy(access); count = 0
        while True:
            choices = [(model == 'titanrtx', name, model, user, part)
                       for name, node in remaining.items() for model in sorted(node['gpus'])
                       if node_slots(node, model, demand)
                       for user in users for part in sorted(node['partitions'])
                       if (part != 'svl' or model == 'titanrtx')
                       and (user, part) in quotas and quotas[(user, part)]['slots'] > 0
                       and quotas[(user, part)]['model_slots'].get(model, 0) > 0]
            if not choices:
                return count
            _, name, model, user, part = min(choices)
            node = remaining[name]; node['gpus'][model] -= 1; node['gpu_total'] -= 1
            node['cpus'] -= demand['cpus']; node['memory_mb'] -= demand['memory_mb']
            quotas[(user, part)]['slots'] -= 1; quotas[(user, part)]['model_slots'][model] -= 1
            count += 1
    accounts = {user: assignable([user]) for user in USERS}
    total = assignable(USERS)
    return {'schema': 'isg-consumed-preparation-cluster-capacity-v1', 'status': 'observed',
            'observed_at': observation['observed_at'], 'accounts_checked': list(USERS),
            'compatible_available_slots': total+held, 'preparation_slots': total, 'account_slots': accounts,
            'critical_titan_reserve_policy': RESERVE_TITANS, 'critical_titan_slots_actually_held': held,
            'free_titan_slots_before_reserve': titan, 'demand': demand,
            'routes': list(access.values()), 'probe': observation, '_reservations': list(reservations),
            'pending_job_ids': [row[6] for row in observation['jobs'] if row[2] == 'PENDING']}


def probe():
    try:
        return build_capacity(observe())
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        return {'status': 'unavailable', 'observed_at': time.time(), 'accounts_checked': list(USERS),
                'account_slots': {}, 'compatible_available_slots': 0, 'preparation_slots': 0, 'error': str(error)}


def reserve_unobserved_receipts(root, capacity):
    """Bridge scheduler visibility lag under the existing shared allocation lock.

    An ambiguous submission stays reserved until reconciled; it is never aged
    out automatically. A submitted job absent from squeue is released only when
    sacct confirms a terminal state. This creates no new claim or owner.
    """
    from scripts.campaign_20261110 import coordinator_dispatch as coordinator
    root = Path(root); observed = {row[6] for row in capacity['probe']['jobs']}
    paths = set()
    policy = coordinator.read(root/'infra/coordinator_dispatch/policy.json')
    for entry in policy.get('preparation_scope', {}).get('normalized_manifests', []):
        reference = entry['manifest']
        coordinator.verify_ref(reference)
        paths.update(Path(t['submission_path']) for t in coordinator.read(reference['path'])['tickets']
                     if t.get('workstream') == 'training preparation')
    returned = root/'vast/consumed_cluster_return'
    paths.update((returned/'submissions').glob('*.json'))
    paths.update(p for p in (returned/'submission_intents').glob('*.json')
                 if not (returned/'submissions'/p.name).exists())
    reservations = {}; terminal = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
                                  'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
    candidates = []
    for path in paths:
        if not path.exists():
            continue
        receipt = coordinator.read(path); routing = receipt.get('preparation_route')
        if not routing:
            continue
        jid = str(receipt.get('job_id') or ''); identity = jid or str(path)
        if jid in observed:
            continue
        candidates.append((jid, identity, routing['route']))
    ids = sorted({jid for jid, _, _ in candidates if jid}); states = {}
    if ids:
        status = subprocess.run(['sacct', '-X', '-n', '-P', '-j', ','.join(ids), '--format=JobIDRaw,State'],
                                capture_output=True, text=True, timeout=15, check=True)
        for line in status.stdout.splitlines():
            fields = line.split('|')
            if len(fields) >= 2 and fields[1]:
                states.setdefault(fields[0], []).append(fields[1].split()[0].rstrip('+'))
    for jid, identity, route in candidates:
        if jid in states and all(state in terminal for state in states[jid]):
            continue
        reservations[identity] = route
    capacity.setdefault('_reservations', []).extend(reservations.values())
    capacity['unobserved_route_receipts'] = list(reservations)
    return capacity


def route_ticket(ticket, capacity, account, *, now=None):
    require(account in USERS and ticket['expected_account'] == account, 'Preparation user identity differs')
    require(ticket.get('workstream') == 'training preparation', 'Only consumed preparation is routed')
    options, payload, values, demand = sbatch_parts(ticket['command'])
    require(capacity.get('status') == 'observed', 'Fresh capacity unavailable')
    reservations = capacity.get('_reservations', [])
    fresh = build_capacity(capacity['probe'], demand, now=now, reservations=reservations)
    access = {(r['user'], r['partition']): r for r in fresh['routes']}
    nodes = inventory(capacity['probe'], demand, reservations, values.get('--exclude', '').split(','))
    reserve_titans(nodes)
    choices = []
    for node in nodes.values():
        for partition in sorted(node['partitions']):
            permitted = access.get((account, partition))
            if not permitted or not permitted['slots']:
                continue
            for model in sorted(node['gpus']):
                if partition == 'svl' and model != 'titanrtx':
                    continue
                free = node_slots(node, model, demand)
                free = min(free, permitted['model_slots'].get(model, 0))
                if free:
                    choices.append((model == 'titanrtx', -free, node['node'], model, permitted))
    if not choices:
        return None
    _, _, node, model, permitted = sorted(choices, key=lambda v: v[:4])[0]
    kept = [arg for name, tokens in options if name not in ROUTING for arg in tokens]
    excluded = sorted(set(EXCLUDED) | set(filter(None, values.get('--exclude', '').split(','))))
    command = ['sbatch', *kept, '--account='+permitted['slurm_account'], '--partition='+permitted['partition'],
               '--gres=gpu:'+model+':1', '--nodelist='+node, '--exclude='+','.join(excluded),
               '--nice='+str(max(1000, int(values.get('--nice', '0')))), *payload]
    # Parse again to prove the executable suffix and CPU/memory demand did not change.
    _, after_payload, _, after_demand = sbatch_parts(command)
    require(payload == after_payload and demand == after_demand, 'Native payload or resources changed')
    return {'command': command, 'prepared_command': ticket['command'], 'payload_sha256': digest(payload),
            'route': {'node': node, 'gpu_model': model, 'partition': permitted['partition'],
                      'slurm_account': permitted['slurm_account'], 'user': account, **demand},
            'capacity_observed_at': fresh['observed_at'], 'query_sha256': capacity['probe']['query_sha256'],
            'critical_titan_slots_actually_held': fresh['critical_titan_slots_actually_held']}


@contextmanager
def allocation_lock(root):
    with (Path(root)/'infra/coordinator_dispatch/capacity_allocation.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def reconcile(root, account, capacity_probe=probe, runner=subprocess.run):
    """Invoked once by an existing root reconciler; never creates a new owner."""
    from scripts.campaign_20261110 import coordinator_dispatch as coordinator
    root = Path(root); policy = coordinator.read(root/'infra/coordinator_dispatch/policy.json')
    require(policy.get('owner') == '/root' and policy.get('production_released') is False, 'Root policy differs')
    tickets = []
    for entry in policy.get('preparation_scope', {}).get('normalized_manifests', []):
        reference = entry['manifest']
        coordinator.verify_ref(reference)
        tickets.extend({**ticket, '_prepared_manifest': reference} for ticket in coordinator.read(reference['path'])['tickets']
                       if ticket.get('workstream') == 'training preparation' and ticket.get('expected_account') == account)
    result = {'observed_at': time.time(), 'submitted': [], 'errors': [], 'owner': '/root', 'production_released': False}
    with allocation_lock(root):
        capacity = capacity_probe(); began = time.monotonic()
        if capacity.get('status') == 'observed':
            reserve_unobserved_receipts(root, capacity)
        else:
            result['status'] = 'blocked_capacity_unknown'
            result['errors'].append({'owner': '/root', 'stage': 'capacity_probe',
                                     'error': capacity.get('error', 'Fresh cluster capacity unavailable'),
                                     'capacity_status': capacity.get('status', 'unknown')})
        for ticket in coordinator.dispatch_order(tickets):
            if capacity.get('status') != 'observed':
                break
            if time.monotonic()-began > 12:
                break
            if coordinator.pending_status(ticket) != 'ready':
                continue
            try:
                coordinator.authorize_preparation(ticket, policy)
                routing = route_ticket(ticket, capacity, account)
                if routing is None:
                    continue
                routed = {**ticket, 'command': routing['command'], 'source_prepared_command': ticket['command'],
                          'preparation_route': routing}
                receipt = coordinator.submit(routed, capacity, runner)
                if receipt:
                    capacity.setdefault('_reservations', []).append(routing['route'])
                    result['submitted'].append({'id': ticket['id'], 'job_id': receipt.get('job_id'),
                                                'status': receipt['status'], 'route': routing['route']})
                    if receipt['status'] != 'submitted':
                        break
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
                result['errors'].append({'id': ticket['id'], 'error': str(error)})
                # An ambiguous external effect retains its claim/intent and ends this batch.
                if coordinator.pending_status(ticket) != 'ready':
                    break
        result['capacity'] = capacity
    coordinator.shared_json(root/'infra/coordinator_dispatch'/('prep_routes_'+account+'.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--account', required=True, choices=USERS); args = parser.parse_args()
    require(getpass.getuser() == args.account, 'Account execution owner differs')
    result = reconcile(args.root, args.account)
    print(json.dumps({key: value for key, value in result.items() if key != 'capacity'}))


if __name__ == '__main__':
    main()
