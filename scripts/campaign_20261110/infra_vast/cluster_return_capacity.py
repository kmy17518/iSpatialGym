"""Live, exact map-worker capacity; a periodic inventory is not a reservation."""
import time

from scripts.campaign_20261110 import dispatch_cluster_first as fresh
from scripts.campaign_20261110.infra_dispatch import backend_capacity


def assess(observation, *, now=None):
    now = time.time() if now is None else now
    result = {'observed_at': observation['observed_at'], 'checked_at': now,
              'probe': observation, 'accounts_checked': ['minyeong', 'shiyuc'],
              'compatible_available_slots': 0, 'account_slots': {}, 'routes': []}
    if observation['state'] != 'available' or not 0 <= now-observation['observed_at'] <= 60:
        return {**result, 'status': 'unavailable_or_stale'}
    demand = {'cpus': 8, 'memory_mb': 48*1024, 'excluded_nodes': ['svl9']}
    nodes = [n for n in observation['nodes'] if 'gres/gpu:titanrtx=' in n.get('CfgTRES', '')]
    physical = {}
    for account in result['accounts_checked']:
        headroom = 0
        for partition in ('svl', 'viscam'):
            backend = account+'_'+partition
            pending = [r[6] for r in observation['jobs'] if r[0] == account
                       and partition in r[1].split(',') and r[2] == 'PENDING'
                       and fresh.gpu_count(r[3]) and (r[5].strip('()') in ('Priority', 'Resources')
                       or r[5].strip('()').startswith(('QOSMax', 'AssocMax')))]
            quota = backend_capacity(backend, observation['jobs'])
            available = {n['NodeName']: fresh.slots(n, partition, demand) for n in nodes}
            usable = min(quota, sum(available.values())) if not pending else 0
            result['routes'].append({'backend': backend, 'slots': usable, 'pending': pending,
                                     'quota_headroom': quota, 'nodes': available})
            if usable:
                headroom += usable
                for name, count in available.items():
                    physical[name] = max(physical.get(name, 0), count)
        result['account_slots'][account] = headroom
    result.update(status='observed', compatible_available_slots=min(sum(physical.values()),
                  sum(result['account_slots'].values())))
    return result


def probe():
    return assess(fresh.probe())
