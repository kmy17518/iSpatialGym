"""Sanitized provider observations and rental-wide GPU time, including idle time."""
import hashlib
import json
from pathlib import Path
import time


def observe(root, instances, complete, started, completed):
    path = Path(root) / 'ledger/rental_presence.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {'schema': 'isg-private-rental-presence-v1', 'started_at_unix': started,
             'observed_at_unix': completed, 'complete': complete,
             'instances': [{'instance': str(row['id']), 'gpu_count': int(row['num_gpus']) if row.get('num_gpus') else None,
                            'status': row.get('actual_status')} for row in instances]}
    with path.open('a') as stream: stream.write(json.dumps(value) + '\n')


def lines(path):
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text().splitlines(keepends=True) if line.strip() and line.endswith('\n')]


def union_seconds(intervals):
    total = 0.; previous = None
    for start, end in sorted(intervals):
        if previous is None: previous = [start, end]
        elif start <= previous[1]: previous[1] = max(previous[1], end)
        else: total += previous[1] - previous[0]; previous = [start, end]
    return total + (previous[1] - previous[0] if previous else 0)


def snapshot(root, timestamp=None):
    root = Path(root); timestamp = time.time() if timestamp is None else timestamp
    events_path = root / 'vast/ledger/events.jsonl'; presence_path = root / 'vast/ledger/rental_presence.jsonl'
    events = lines(events_path); observations = lines(presence_path)
    complete = [row for row in observations if row['complete']]
    epoch = complete[0]['observed_at_unix'] if complete else None
    observed = complete[-1]['observed_at_unix'] if complete else None
    created, ended, counts = {}, {}, {}
    for row in events:
        iid = str(row.get('instance'))
        if row['ev'] == 'create':
            created[iid] = row
            count = row.get('num_gpus') or (len(row['gpu_ids']) if row.get('gpu_ids') else None)
            if count: counts[iid] = int(count)
        elif row['ev'] == 'gpu_identity_backfill' and iid in created:
            if row.get('offer') == created[iid].get('offer') and row.get('machine_id') == created[iid].get('machine_id'):
                counts[iid] = len(row['gpu_ids'])
        elif row['ev'] == 'destroy' and row.get('ok'): ended.setdefault(iid, row)
    for row in observations:
        for item in row['instances']:
            iid, count = item['instance'], item['gpu_count']
            if count is None: continue
            if iid in counts and counts[iid] != count: raise ValueError('rental GPU count changed: ' + iid)
            counts[iid] = count
    attempts, missing = {}, []
    for path in sorted((root/'vast/state/attempts').glob('*/*.json')):
        value = json.loads(path.read_text()); done = value['done']; iid = str(value.get('instance'))
        start, end, count = done.get('fetch_started_unix'), done.get('upload_finished_unix'), done.get('gpu_count')
        if start is None or end is None or count is None:
            missing.append(str(path)); continue
        if end < start or int(count) <= 0: raise ValueError('invalid worker accounting interval')
        if iid in counts and counts[iid] != int(count): raise ValueError('worker/rental GPU count differs')
        attempts.setdefault(iid, []).append((start, end))
    records=[]; window_unknown=[]; window_total=0.; window_attributed=0.
    for iid, event in created.items():
        end = min(timestamp, ended.get(iid, {}).get('time', timestamp)); start = event['time']; count = counts.get(iid)
        record = {'instance': iid, 'gpu_count': count, 'created_observation_unix': start,
                  'destroyed_observation_unix': ended.get(iid, {}).get('time'),
                  'rental_observation_gpu_seconds': max(0, end-start)*count if count else None,
                  'creation_request_started_unix': event.get('request_started_unix'),
                  'creation_response_completed_unix': event.get('response_completed_unix')}
        if epoch is not None and end > epoch:
            first, last = max(start, epoch), min(end, observed)
            if last > first:
                if count is None: window_unknown.append(iid)
                else:
                    total = (last-first)*count
                    intervals = [(max(a,first),min(b,last)) for a,b in attempts.get(iid,[]) if min(b,last)>max(a,first)]
                    attributed = union_seconds(intervals)*count
                    record.update(window_gpu_seconds=total, window_attributed_gpu_seconds=attributed,
                                  window_unattributed_gpu_seconds=total-attributed)
                    window_total += total; window_attributed += attributed
        records.append(record)
    orphan_presence=sorted({item['instance'] for row in observations for item in row['instances']} - set(created))
    unavailable = not complete or bool(window_unknown or orphan_presence)
    return {'schema':'isg-vast-rental-gpu-accounting-v1','observed_at_unix':timestamp,
            'provider_observed_at_unix':observed,'instrumentation_epoch_unix':epoch,
            'fleet_window_estimated_gpu_seconds':None if unavailable else window_total,
            'fleet_window_measured_attributed_worker_gpu_seconds':window_attributed,
            'fleet_window_unattributed_gpu_seconds':None if unavailable else window_total-window_attributed,
            'window_unknown_gpu_count_instances':window_unknown,'provider_instances_without_creation':orphan_presence,
            'historical_unknown_gpu_count_instances':[row['instance'] for row in records if row['gpu_count'] is None],
            'attempts_missing_complete_intervals':missing,'rentals':records,
            'sources':[{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                       for path in (events_path,presence_path) if path.exists()],
            'boundary_precision':'Provider observations, not exact billing boundaries. Creation request/response times are retained prospectively; disappearance is bounded by polling. Legacy absent timing is not reconstructed.',
            'scope':'All campaign Vast rentals after the instrumentation epoch, including boot, startup, failed attempts, retries, transfers, replay and idle. Attributed time is the union of retained worker fetch-to-upload intervals; remaining time is deliberately unattributed.'}
