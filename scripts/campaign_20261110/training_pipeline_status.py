"""Read-only projection of bounded pipeline attempts; never author or submit work."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from scripts.campaign_20261110.contracts import file_digest, require, write_json_atomic


def read(path): return json.loads(Path(path).read_text())
def ref(path): return {'path': str(path), 'sha256': file_digest(path)}
def timestamp(path): return datetime.fromtimestamp(Path(path).stat().st_mtime, timezone.utc).isoformat()


def observe(root, *, now=None):
    root = Path(root).resolve(); base = root/'training_pipeline_canaries'
    result = {'schema': 'isg-training-pipeline-status-v1', 'campaign_id': 'campaign_20261110',
        'observed_at': now or datetime.now(timezone.utc).isoformat(), 'records': [], 'batch_stages': [],
        'sources': [], 'errors': [], 'benchmark_credit': 0, 'quota_consumed': 0,
        'production_released': False, 'status': 'observed'}
    reviews = {}
    for path in sorted(base.glob('*/existing_native_reviews.json')):
        try:
            value = read(path)
            require(value['campaign_id'] == result['campaign_id'], 'foreign review observation')
            result['sources'].append({**ref(path), 'observed_at': value.get('observed_at'), 'kind': 'existing_native_reviews'})
            for candidate, row in value['records'].items():
                allocation = row.get('allocation') or {}
                start = allocation.get('start')
                if start is not None:
                    reviews[(candidate, int(start))] = (row, value.get('observed_at'), ref(path))
        except Exception as error:
            result['errors'].append({'path': str(path), 'error': f'{type(error).__name__}: {error}'})
    for path in sorted(base.glob('*/requests.json')):
        try:
            value = read(path)
            require(value['campaign_id'] == result['campaign_id'], 'foreign canary request')
            for request in value['records']:
                key = (request['candidate_id'], int(request['start']))
                row = {'candidate_id': key[0], 'start': key[1], 'count': request['count'],
                    'class_id': request['class_id'], 'configuration_sha256': request['config_sha256'],
                    'native_runtime': request['native_runtime'], 'batch': path.parent.name,
                    'request': ref(path), 'status': 'native_not_observed', 'native': None,
                    'review': None, 'benchmark_credit': 0, 'quota_consumed': 0}
                output = Path(request['output']).resolve()
                require(output.is_relative_to(base), 'native output outside pipeline scope')
                gate_path = output/'sampling_gate.json'
                if gate_path.exists():
                    gate = read(gate_path)
                    require(gate['candidate_id'] == key[0] and gate['config_sha256'] == request['config_sha256'],
                            'native gate configuration differs')
                    row['native'] = {**ref(gate_path), 'observed_at': timestamp(gate_path),
                        'returncode': gate.get('returncode'), 'written': gate.get('tro_count'),
                        'elapsed_seconds': gate.get('elapsed_seconds')}
                    row['status'] = 'native_passed' if gate.get('returncode') == 0 and gate.get('tro_count') == 1 else 'native_failed'
                if key in reviews:
                    record, observed_at, source = reviews[key]
                    row['review'] = {**record, 'observed_at': observed_at, 'observation_source': source}
                    row['status'] = record['status']
                    if record.get('validation_review'):
                        validation = record['validation_review']; review_path = Path(validation['review']).resolve()
                        require(review_path.is_relative_to(root) and file_digest(review_path) == validation['sha256'],
                                'canary review changed')
                        proof = read(review_path)
                        require(proof['validation_passed'] is True and proof['accepted'] is False and
                                proof['quota_consumed'] == 0, 'review is not zero-credit canary evidence')
                result['records'].append(row)
        except Exception as error:
            result['errors'].append({'path': str(path), 'error': f'{type(error).__name__}: {error}'})
    for request in sorted(base.glob('*/batch_restoration*/request.json')):
        batch = {'request': ref(request), 'batch': str(request.parent.relative_to(base)),
                 'observed_at': timestamp(request), 'status': 'prepared_unvalidated', 'stages': [],
                 'benchmark_credit': 0, 'quota_consumed': 0}
        failure = request.parent/'preflight_failure.json'
        if failure.exists(): batch.update(status='preflight_failed', failure={**ref(failure), **read(failure)})
        elif (request.parent/'preflight.json').exists(): batch['status'] = 'ready_for_bounded_probe'
        submission_path = request.parent/'submission.json'
        submissions = {}
        if submission_path.exists():
            try:
                submitted = read(submission_path)
                require(submitted['request'] == ref(request), 'batch submission request differs')
                submissions = {job['kind']: job for job in submitted['jobs']}
                batch['submission'] = ref(submission_path)
            except Exception as error:
                batch.update(status='submission_observation_failed', error=f'{type(error).__name__}: {error}')
        for name, kind in [('physics', 'physics_action_open_loop'), ('render', 'kinematic_observation_rendering')]:
            probe = request.parent/name/'probe.json'
            stage = {'kind': kind, 'status': 'not_observed'}
            if kind in submissions:
                stage.update(status='submitted', allocation=submissions[kind], scheduler_state='unavailable')
            if probe.exists():
                stage.update(evidence=ref(probe), observed_at=timestamp(probe), status='incomplete')
                try:
                    native = read(probe); stage['episodes_observed'] = len(native.get('entries', []))
                    if stage['episodes_observed'] == 3:
                        from scripts.campaign_20261110.training_restoration import verify_batch
                        verify_batch(root, ref(probe), 1794321600, kind)
                        stage['status'] = 'verified'
                except Exception as error:
                    stage.update(status='failed_validation', error=f'{type(error).__name__}: {error}')
            batch['stages'].append(stage)
        if all(s['status'] == 'verified' for s in batch['stages']): batch['status'] = 'batch_restoration_verified'
        result['batch_stages'].append(batch)
    keys = [(r['candidate_id'], r['start']) for r in result['records']]
    require(len(keys) == len(set(keys)), 'duplicate exact canary attempt in projection')
    result['counts'] = dict(Counter(r['status'] for r in result['records']))
    result['attempts_observed'] = len(result['records'])
    if result['errors']: result['status'] = 'partial_observation'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--interval', type=float, default=0)
    args = parser.parse_args(); output = args.root/'training_pipeline_canaries/pipeline_status.json'
    while True:
        value = observe(args.root)
        write_json_atomic(output, value)
        print(json.dumps({'observed_at': value['observed_at'], 'attempts': value['attempts_observed'],
            'errors': len(value['errors']), 'benchmark_credit': 0}), flush=True)
        if args.interval <= 0: break
        time.sleep(args.interval)


if __name__ == '__main__': main()
