"""Read-only, continuously refreshed forecasts for the three release gates."""
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def checked(ref):
    raw = Path(ref['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref['sha256']:
        raise ValueError('Joint schedule source changed: '+ref['path'])
    return json.loads(raw)


def quantile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered)-1, math.ceil((len(ordered)-1)*fraction))]


def forecast(remaining, seconds, concurrency, now, overhead=(600, 1200)):
    if not seconds or concurrency <= 0:
        return None
    return [now+remaining*quantile(seconds, q)/concurrency+delay
            for q, delay in zip((.5, .9), overhead)]


def project(root, flow, now):
    root = Path(root); path = root/'infra/coordination/joint_schedule_inputs.json'
    if not path.exists():
        return {'status': 'unavailable', 'observed_at': now, 'common_finish_range': None,
                'preparation_extra_delay_seconds': None}
    spec = read(path); errors = []; samples = []; keys = set()
    for wave in spec['preparation_waves']:
        chunks = checked(wave['index'])['chunks'] if 'index' in wave else [{'manifest': wave['manifest']}]
        for chunk in chunks:
            keys.update(t['physical_dedup_key'] for t in checked(chunk['manifest'])['jobs'])
        for status in Path(wave['reader_directory']).glob(wave.get('reader_glob', '*/status.json')):
            for row in read(status).get('rows', []):
                elapsed = row.get('builder_elapsed_seconds')
                if row.get('status') in ('verified_positive', 'verified_zero') and isinstance(elapsed, (int, float)) and elapsed > 0:
                    samples.append(elapsed)
    items = {r['physical_dedup_key']: r for r in flow.get('items', []) if r['physical_dedup_key'] in keys}
    missing = keys-items.keys()
    if missing:
        errors.append({'kind': 'joint_schedule_wave_observation_incomplete', 'count': len(missing), 'owner': '/root'})
    remaining = [r for r in items.values() if not r.get('verified_at') and not r.get('awaiting_verification')
                 and r.get('verification_status') != 'failed']
    failed = sum(r.get('verification_status') == 'failed' for r in items.values())
    metrics = read(root/'vast/metrics.json')
    fresh = 0 <= now-metrics.get('observed_at_unix', 0) <= 180
    all_items = {r['physical_dedup_key']: r for r in flow.get('items', [])}
    ids = {str(r.get('job_id')) for r in items.values()}
    all_ids = {str(r.get('job_id')) for r in all_items.values()}
    heartbeat_jobs = {str(j) for w in metrics.get('instances', []) if fresh and w.get('worker_observation_current')
                      for j in w.get('heartbeat_running_jobs', [])}
    vast_running = {str(j) for w in metrics.get('instances', []) if fresh and w.get('worker_observation_current')
                    for j in w.get('heartbeat_running_jobs', []) if str(j) in ids}
    cluster_running = sum(r.get('backend') != 'vast' and r.get('execution_state') == 'RUNNING' for r in items.values())
    current = cluster_running+len(vast_running)
    all_cluster_running = sum(r.get('backend') != 'vast' and r.get('execution_state') == 'RUNNING'
                              for r in all_items.values())
    all_current = all_cluster_running+len(heartbeat_jobs & all_ids)
    other_current = max(0, all_current-current)
    desired = int(spec['desired_preparation_gpus'])
    desired_maps = min(max(0, desired-other_current), len(remaining))
    conditional = [{'useful_gpus': c, 'known_wave_finish_range': forecast(len(remaining), samples, c, now)}
                   for c in sorted({c for c in (current, min(100, len(remaining)), desired_maps) if c > 0})]
    streams = spec['streams']
    ranges = [r.get('finish_range') for r in streams]
    complete = len(ranges) == 3 and all(r and len(r) == 2 for r in ranges)
    common = [max(r[i] for r in ranges) for i in (0, 1)] if complete else None
    prep = next((r.get('finish_range') for r in streams if r['id'] == 'preparation'), None)
    peers = [r.get('finish_range') for r in streams if r['id'] != 'preparation']
    target = [max(r[i] for r in peers) for i in (0, 1)] if len(peers) == 2 and all(peers) else None
    delay = [max(0, prep[0]-target[1]), max(0, prep[1]-target[0])] if prep and target else None
    if delay and delay[1] > 0:
        errors.append({'kind': 'preparation_forecast_tail', 'owner': '/root', 'delay_seconds': delay})
    if now-spec['observed_at'] > 300:
        errors.append({'kind': 'joint_engineering_forecast_review_due', 'owner': '/root', 'last_review_at': spec['observed_at']})
    if any(r.get('engineering_blockers') for r in streams):
        errors.append({'kind': 'joint_readiness_engineering_blocker', 'owner': '/root',
                       'workstreams': [r['id'] for r in streams if r.get('engineering_blockers')]})
    exposure = sum(metrics[k] for k in ('spend_estimate_usd', 'future_compute_commitment_usd', 'transfer_reserve_usd'))
    return {'schema': 'isg-joint-readiness-forecast-v1', 'observed_at': now, 'root_review_at': spec['observed_at'],
            'streams': streams, 'common_finish_range': common, 'preparation_target_range': target,
            'preparation_extra_delay_seconds': delay, 'full_preparation_forecast_known': prep is not None,
            'known_wave': {'physical_jobs': len(keys), 'observed_jobs': len(items), 'remaining_gpu_jobs': len(remaining),
                           'awaiting_verification': sum(r.get('awaiting_verification', False) for r in items.values()),
                           'failed_needing_diagnosis': failed, 'timing_sample_count': len(samples),
                           'native_seconds_p50': quantile(samples, .5) if samples else None,
                           'native_seconds_p90': quantile(samples, .9) if samples else None,
                           'remaining_native_gpu_hours_range': [len(remaining)*quantile(samples,q)/3600 for q in (.5,.9)] if samples else None,
                           'conditional_finish_ranges': conditional,
                           'assumption': 'Measured native p50-p90 planning scenarios plus10-20minutes for startup/transfer/verification. Not a confidence interval or full preparation ETA; failed items require repair.'},
            'capacity': {'desired_preparation_gpus': desired, 'cluster_running_wave_jobs': cluster_running,
                         'vast_heartbeat_running_wave_jobs': len(vast_running), 'current_executing_wave_jobs': current,
                         'current_executing_preparation_jobs': all_current,
                         'cluster_running_preparation_jobs': all_cluster_running,
                         'vast_heartbeat_running_preparation_jobs': len(heartbeat_jobs & all_ids),
                         'other_required_preparation_executing': other_current,
                         'desired_known_wave_gpus_after_other_work': desired_maps,
                         'vast_provider_gpus': metrics.get('running_gpus') if fresh else None,
                         'additional_useful_gpus_needed': max(0, desired-all_current), 'gpu_utilization': 'unmeasured',
                         'vast_observation_current': fresh, 'inventory_complete': False},
            'budget': {'total_cap_usd': 3000, 'observed_exposure_usd': exposure, 'remaining_after_commitments_usd': 3000-exposure},
            'resource_actions': spec['resource_actions'], 'exceptions': errors, 'production_released': False}


def publish(root, flow, now):
    value = project(root, flow, now); path = Path(root)/'infra/joint_readiness_forecast.json'
    temporary = path.with_suffix('.json.new'); temporary.write_text(json.dumps(value, indent=2)+'\n'); temporary.replace(path)
    return value
