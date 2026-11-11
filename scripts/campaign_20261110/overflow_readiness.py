"""Fresh cluster and funded-demand evidence before cancelling pending work."""
import json
import math
from pathlib import Path
import time

from scripts.campaign_20261110.contracts import file_digest, require


def pending_delay(value):
    require(math.isfinite(value) and value >= 0, 'pending delay must be finite and nonnegative')
    return value


def check(root, scheduler, *, now=None):
    root = Path(root)
    now = time.time() if now is None else now
    reason = scheduler.get('Reason', '')
    require(reason in ('Priority', 'Resources') or reason.startswith(('QOSMax', 'AssocMax')),
            'pending work is not independently resource-ready: ' + reason)
    paths = {name: root / relative for name, relative in {
        'capacity': 'infra/capacity_recovery/ledger.json',
        'metrics': 'vast/metrics.json', 'config': 'vast/config.json'}.items()}
    data = {name: json.loads(path.read_text()) for name, path in paths.items()}
    require(0 <= now - data['capacity']['observed_at'] <= 180, 'fresh cluster capacity required')
    require(0 <= now - data['metrics']['observed_at_unix'] <= 300, 'fresh funded budget required')
    cfg, metrics = data['config'], data['metrics']
    budget = float(cfg['budget_usd'])
    require(0 < budget <= 3000, 'campaign budget exceeds authorization')
    rate = float(cfg['max_dph_gpu'])
    require(math.isfinite(rate) and rate > 0, 'invalid maximum hourly reservation')
    # Reserve every activated queue entry conservatively, including entries not
    # yet materialized by the coordinator. Existing worker commitments remain
    # counted separately; this may overcount but never spends an assumed refund.
    queued = 0.0
    for jid in set(map(str, cfg.get('dispatch_job_ids', []))):
        state = root / 'vast/state/jobs' / (jid + '.json')
        queue = root / 'vast/queue' / (jid + '.json')
        if state.exists():
            row = json.loads(state.read_text())
            if row.get('status') not in ('queued', 'assigned'):
                continue
        elif queue.exists():
            row = json.loads(queue.read_text())
        else:
            continue
        duration = float(row.get('time_s', 10800))
        require(math.isfinite(duration) and duration > 0, 'invalid queued runtime reservation')
        queued += duration / 3600 * rate
    limit = scheduler.get('TimeLimit', '')
    require(limit and limit not in ('UNLIMITED', 'Partition_Limit'), 'bounded scheduler time required')
    days, _, clock = limit.partition('-')
    if not clock:
        clock, days = days, '0'
    parts = list(map(int, clock.split(':')))
    require(len(parts) in (2, 3), 'unrecognized scheduler time limit')
    seconds = int(days) * 86400 + (parts[0] * 3600 + parts[1] * 60 + parts[2] if len(parts) == 3 else parts[0] * 60 + parts[1])
    require(seconds > 0, 'positive bounded runtime required')
    amounts = [float(metrics[k]) for k in ('spend_estimate_usd', 'future_compute_commitment_usd', 'transfer_reserve_usd')]
    require(all(math.isfinite(v) and v >= 0 for v in amounts), 'invalid budget observation')
    total = sum(amounts) + queued + seconds / 3600 * rate
    require(total <= budget, 'insufficient funded overflow reservation')
    return {'observed_at': now, 'pending_reason': reason, 'minimum_pending_seconds': 0,
            'sources': {name: {'path': str(path), 'sha256': file_digest(path)} for name, path in paths.items()},
            'maximum_budget_including_queue_and_job': total, 'budget_usd': budget,
            'new_job_reservation_usd': seconds / 3600 * rate, 'queued_reservation_usd': queued,
            'fleet_ceiling': None, 'worker_admission': 'coordinator accounts for compatible existing and warming workers'}
