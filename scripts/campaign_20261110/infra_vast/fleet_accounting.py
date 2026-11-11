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


def exclude_intervals(intervals, exclusions):
    result = list(intervals)
    for left, right in exclusions:
        remaining = []
        for start, end in result:
            if right <= start or left >= end:
                remaining.append((start, end))
            else:
                if start < left: remaining.append((start, left))
                if right < end: remaining.append((right, end))
        result = remaining
    return result


def stopped_disk_intervals(root, observations, timestamp):
    """Only owner-stopped, evidence-preserved disks have non-GPU intervals."""
    path = Path(root)/'infra/training_production_gate/preserved_instances.json'
    if not path.exists(): return {}, None
    body = path.read_bytes(); value = json.loads(body)
    if value.get('campaign_id') != 'campaign_20261110':
        raise ValueError('stopped-disk accounting campaign differs')
    targets = set(value['instances']); opened = {}; intervals = {iid:[] for iid in targets}
    for observation in observations:
        if not observation['complete']: continue
        at = observation['observed_at_unix']
        if at < value.get('created_at', 0): continue
        for row in observation['instances']:
            iid = str(row['instance'])
            if iid not in targets: continue
            if row['status'] in ('exited', 'stopped'):
                opened.setdefault(iid, at)
            elif row['status'] is not None and iid in opened:
                intervals[iid].append((opened.pop(iid), at))
    for iid, start in opened.items(): intervals[iid].append((start, timestamp))
    return intervals, {'path':str(path),'sha256':hashlib.sha256(body).hexdigest(),
                       'boundary':'First complete stopped observation to first resumed observation; exact transition remains bounded by polling.'}


def historical_counts(root, created):
    """Recover counts only from pinned exact offers preceding their rental."""
    root=Path(root).resolve();path=root/'accounting/vast_rental_gpu_count_sources.json'
    if not path.exists():return {},None
    body=path.read_bytes();manifest=json.loads(body)
    if manifest.get('schema')!='isg-vast-exact-historical-gpu-count-v1' or manifest.get('campaign_id')!='campaign_20261110':
        raise ValueError('historical rental count source identity differs')
    counts={};cached={}
    for row in manifest['records']:
        iid=str(row['instance']);event=created.get(iid)
        if not event or event.get('offer')!=row['offer'] or event.get('machine_id')!=row['machine_id']:
            raise ValueError('historical rental allocation identity differs: '+iid)
        source=Path(row['source_path']).resolve()
        if not source.is_relative_to(root/'vast/market'):raise ValueError('historical count source outside campaign market')
        if source not in cached:
            source_body=source.read_bytes();cached[source]=(hashlib.sha256(source_body).hexdigest(),json.loads(source_body))
        digest,market=cached[source]
        if digest!=row['source_sha256'] or market['time']>event['time']:
            raise ValueError('historical count source changed or postdates rental: '+iid)
        matches=[offer for offer in market['offers'] if offer.get('id')==row['offer'] and offer.get('machine_id')==row['machine_id']]
        actual={int(offer['num_gpus']) for offer in matches if offer.get('num_gpus')}
        if actual!={int(row['num_gpus'])} or int(row['num_gpus'])<=0:
            raise ValueError('historical offer GPU count is ambiguous: '+iid)
        counts[iid]=int(row['num_gpus'])
    return counts,{'path':str(path),'sha256':hashlib.sha256(body).hexdigest(),'recovered_rentals':len(counts)}


def snapshot(root, timestamp=None):
    root = Path(root); timestamp = time.time() if timestamp is None else timestamp
    events_path = root / 'vast/ledger/events.jsonl'; presence_path = root / 'vast/ledger/rental_presence.jsonl'
    events = lines(events_path); observations = lines(presence_path)
    complete = [row for row in observations if row['complete']]
    epoch = complete[0]['observed_at_unix'] if complete else None
    observed = complete[-1]['observed_at_unix'] if complete else None
    stopped, stopped_source = stopped_disk_intervals(root, observations, observed or timestamp)
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
    recovered,count_source=historical_counts(root,created)
    for iid,count in recovered.items():
        if iid in counts and counts[iid]!=count:raise ValueError('historical/current rental GPU count differs: '+iid)
        counts[iid]=count
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
    records=[]; window_unknown=[]; window_total=0.; window_attributed=0.;recorded_total=0.;recorded_attributed=0.
    for iid, event in created.items():
        end = min(timestamp, observed if observed is not None else timestamp,
                  ended.get(iid, {}).get('time', timestamp)); start = event['time']; count = counts.get(iid)
        gpu_intervals = exclude_intervals([(start,end)], stopped.get(iid, [])) if end > start else []
        gpu_seconds = union_seconds(gpu_intervals)
        record = {'instance': iid, 'gpu_count': count, 'created_observation_unix': start,
                  'destroyed_observation_unix': ended.get(iid, {}).get('time'),
                  'rental_observation_gpu_seconds': gpu_seconds*count if count else None,
                  'disk_only_observed_seconds': max(0,end-start)-gpu_seconds,
                  'disk_only_observation_intervals': stopped.get(iid, []),
                  'creation_request_started_unix': event.get('request_started_unix'),
                  'creation_response_completed_unix': event.get('response_completed_unix')}
        if count is not None:
            total=gpu_seconds*count
            attributed=union_seconds(exclude_intervals([(max(a,start),min(b,end)) for a,b in attempts.get(iid,[]) if min(b,end)>max(a,start)], stopped.get(iid, [])))*count
            recorded_total+=total;recorded_attributed+=attributed
            record.update(recorded_attributed_gpu_seconds=attributed,recorded_unattributed_gpu_seconds=total-attributed)
        if epoch is not None and end > epoch:
            first, last = max(start, epoch), min(end, observed)
            if last > first:
                if count is None: window_unknown.append(iid)
                else:
                    total = union_seconds(exclude_intervals([(first,last)], stopped.get(iid, [])))*count
                    intervals = [(max(a,first),min(b,last)) for a,b in attempts.get(iid,[]) if min(b,last)>max(a,first)]
                    attributed = union_seconds(exclude_intervals(intervals, stopped.get(iid, [])))*count
                    record.update(window_gpu_seconds=total, window_attributed_gpu_seconds=attributed,
                                  window_unattributed_gpu_seconds=total-attributed)
                    window_total += total; window_attributed += attributed
        records.append(record)
    orphan_presence=sorted({item['instance'] for row in observations for item in row['instances']} - set(created))
    unavailable = not complete or bool(window_unknown or orphan_presence)
    historical_unknown=[row['instance'] for row in records if row['gpu_count'] is None]
    recorded_unavailable=not complete or bool(historical_unknown or orphan_presence)
    return {'schema':'isg-vast-rental-gpu-accounting-v1','campaign_id':'campaign_20261110','observed_at_unix':timestamp,
            'provider_observed_at_unix':observed,'instrumentation_epoch_unix':epoch,
            'fleet_window_estimated_gpu_seconds':None if unavailable else window_total,
            'fleet_window_measured_attributed_worker_gpu_seconds':window_attributed,
            'fleet_window_unattributed_gpu_seconds':None if unavailable else window_total-window_attributed,
            'recorded_fleet_started_at_unix':min((event['time'] for event in created.values()),default=None),
            'recorded_fleet_estimated_gpu_seconds':None if recorded_unavailable else recorded_total,
            'recorded_fleet_measured_attributed_worker_gpu_seconds':recorded_attributed,
            'recorded_fleet_unattributed_gpu_seconds':None if recorded_unavailable else recorded_total-recorded_attributed,
            'historical_count_recovery':count_source,
            'stopped_disk_accounting':stopped_source,
            'disk_only_observed_seconds':sum(row['disk_only_observed_seconds'] for row in records),
            'window_unknown_gpu_count_instances':window_unknown,'provider_instances_without_creation':orphan_presence,
            'historical_unknown_gpu_count_instances':historical_unknown,
            'attempts_missing_complete_intervals':missing,'rentals':records,
            'sources':[{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                       for path in (events_path,presence_path) if path.exists()],
            'boundary_precision':'Provider observations, not exact billing boundaries. Creation request/response times are retained prospectively; disappearance is bounded by polling. Legacy absent timing is not reconstructed.',
            'scope':'All recorded campaign Vast GPU rentals, including boot, startup, failed attempts, retries, transfers, replay and idle. Verified owner-stopped disk-retention intervals are excluded using provider observations; recovery restart time is included again. The fleet_window fields retain the prospective observation epoch; recorded_fleet fields additionally cover historical rentals only when their exact GPU counts are proven. Attributed time is the union of retained worker fetch-to-upload intervals; remaining time is deliberately unattributed.'}
