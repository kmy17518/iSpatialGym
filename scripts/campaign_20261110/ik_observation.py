"""Read-only exact-IK evidence audit. Never transfers caches or controls jobs."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import time

CAMPAIGN = 'campaign_20261110'
ACCOUNTS = ('minyeong', 'shiyuc', 'vast')
PRIMARY_ROOT = Path('/vision/u/minyeong/projects/isg_home/campaign_20261110')
SECONDARY_ROOT = Path('/vision/u/shiyuc/minyeong/campaign_20261110')
METRICS = ('exact_queries', 'exact_cache_eligible_queries', 'exact_cache_candidate_queries',
           'exact_verified_hit', 'exact_shared_verified_hit', 'exact_rejected_cache_queries',
           'exact_fresh_queries', 'exact_fresh_solver_calls')
ENV_KEYS = ('ISGE_CODE', 'ISPATIALGYM_EXACT_IK_MANIFEST', 'ISPATIALGYM_EXACT_IK_LOCAL_CACHE',
            'ISPATIALGYM_EXACT_IK_METRICS', 'ISPATIALGYM_EXACT_IK_PRODUCER', 'ISGF_SAMPLING_CACHE',
            'ISPATIALGYM_GRASP_LIBRARY_V2', 'ISPATIALGYM_REACH_MAPS', 'ISGF_ASSET_CACHE',
            'KIT_TPL_NFS_ROOT', 'KIT_TEX_STORE')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def owned(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(root.resolve()):
        # Both exact campaign roots are owner-authorized. This reads existing
        # observations only and never changes secondary-account permissions.
        if root.resolve() != PRIMARY_ROOT or not path.is_relative_to(SECONDARY_ROOT):
            raise ValueError('observation path outside campaign')
    return path


def read(path, root):
    path = owned(path, root)
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError('observation exceeds bounded size')
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('expected observation object')
    return value


def env_values(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        key, sep, value = line.partition('=')
        if sep and key in ENV_KEYS:
            values[key] = ' '.join(shlex.split(value))
    return values


def audit(root):
    root = Path(root).resolve()
    if root.name != CAMPAIGN:
        raise ValueError('wrong campaign root')
    base = root / 'ik_exchange'
    rows, errors, seen_metrics = [], [], set()
    for registration in sorted((base / 'registered_jobs').glob('*.json')):
        try:
            reg = read(registration, root)
            if reg.get('campaign_id') != CAMPAIGN or reg.get('account') not in ACCOUNTS:
                raise ValueError('foreign job registration')
            job = owned(reg['job_dir'], root)
            if job.is_relative_to(SECONDARY_ROOT) and reg['account'] != 'shiyuc':
                raise ValueError('secondary job attributed to wrong account')
            env_path = job / 'job.env'
            env = env_values(env_path)
            row = {'job_id': str(reg['job_id']), 'account': reg['account'], 'job_dir': str(job),
                   'job_env_sha256': sha(env_path), 'declared_environment': env,
                   'metrics': [], 'transfer_receipts': [], 'lifecycle_events': [], 'errors': []}
            rows.append(row)
            path = env.get('ISPATIALGYM_EXACT_IK_MANIFEST')
            row['enabled'] = bool(path and env.get('ISPATIALGYM_EXACT_IK_LOCAL_CACHE'))
            manifest = read(path, root) if path else {}
            row['declared_compatibility_id'] = manifest.get('compatibility_sha256')
            metric_root = env.get('ISPATIALGYM_EXACT_IK_METRICS')
            paths = sorted(owned(metric_root, root).glob('*.json')) if metric_root else []
            for path in paths:
                try:
                    value = read(path, root)
                    if value.get('campaign_id') != CAMPAIGN:
                        raise ValueError('foreign runtime metric')
                    if value.get('producer') != env.get('ISPATIALGYM_EXACT_IK_PRODUCER'):
                        raise ValueError('runtime producer differs from registered job')
                    if value.get('compatibility_sha256') not in (None, row['declared_compatibility_id']):
                        raise ValueError('runtime compatibility differs from registered manifest')
                    if str(path.resolve()) in seen_metrics:
                        raise ValueError('metric already attributed to another registered job')
                    counters = value.get('counters', {})
                    if any(type(v) is not int or v < 0 for v in counters.values()):
                        raise ValueError('invalid runtime counter')
                    seen_metrics.add(str(path.resolve()))
                    row['metrics'].append({k: value.get(k) for k in
                        ('observed_at', 'pid', 'producer', 'compatibility_sha256', 'local_cache', 'counters')})
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    row['errors'].append({'path': str(path), 'error': str(exc)})
            for kind in ('import_receipts', 'export_receipts'):
                for name in reg.get(kind, [])[-100:]:
                    try:
                        value = read(name, root)
                        if value.get('campaign_id') != CAMPAIGN:
                            raise ValueError('foreign transfer receipt')
                        row['transfer_receipts'].append({'kind': kind, 'path': name, 'sha256': sha(name),
                            **{k: value.get(k) for k in ('observed_at', 'complete', 'entries', 'branches',
                            'bytes', 'elapsed_seconds', 'errors', 'deferred', 'durable_receipt', 'source')}})
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        row['errors'].append({'path': name, 'error': str(exc)})
            for name in reg.get('lifecycle_paths', [])[-20:]:
                try:
                    value = read(name, root)
                    if value.get('campaign_id') != CAMPAIGN or str(value.get('job_id')) != row['job_id']:
                        raise ValueError('foreign lifecycle observation')
                    row['lifecycle_events'].extend(value.get('events', [])[-100:])
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    row['errors'].append({'path': name, 'error': str(exc)})
            total = Counter()
            for value in row['metrics']:
                total.update(value['counters'])
            row['counters'] = {k: total[k] for k in METRICS} if row['metrics'] else None
            row['runtime_compatibility_verified'] = bool(row['metrics']) and all(
                m['compatibility_sha256'] == row['declared_compatibility_id'] and bool(m['compatibility_sha256'])
                for m in row['metrics'])
            row['actual_local_caches'] = sorted({m['local_cache'] for m in row['metrics'] if m['local_cache']})
            row['last_runtime_observed_at'] = max((m['observed_at'] for m in row['metrics'] if m['observed_at']), default=None)
            row['sampling_time_benefit_seconds'] = None
            row['benefit_reason'] = 'Comparable measured work has not established a sampling-time benefit.'
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({'path': str(registration), 'error': type(exc).__name__ + ': ' + str(exc)})
    accounts = []
    for account in ACCOUNTS:
        selected = [r for r in rows if r['account'] == account]
        measured = [r for r in selected if r.get('counters') is not None]
        counts = Counter()
        for row in measured:
            counts.update(row['counters'])
        accounts.append({'account': account, 'registered_jobs': len(selected), 'measured_jobs': len(measured),
                         'counters': {k: counts[k] for k in METRICS} if measured else None,
                         'status': 'measured_runtime' if measured else 'enabled_without_runtime_evidence' if selected else 'not_observed',
                         'observation_scope': 'Registered instrumented jobs only; unobserved jobs are not inferred to have zero reuse.'})
    return {'schema': 'isg-exact-ik-observation-v1', 'campaign_id': CAMPAIGN,
            'observed_at': datetime.now(timezone.utc).isoformat(), 'accounts': accounts, 'jobs': rows, 'errors': errors,
            'semantics': 'Enabled configuration, validated runtime compatibility, installed candidates and live verified IK hits are separate stages. Shader/texture warming does not establish IK reuse. Missing denominators remain unavailable.',
            'denominators': 'Verified-hit counts are query counts, compared with exact_cache_eligible_queries; fresh_solver_calls count solver invocations and can exceed query counts. Import/merge receipts alone do not prove a verified hit.',
            'roundtrip_status': 'Unproved until cluster-to-cluster import, Vast verified hits, durable return and cluster verified reuse are linked by explicit evidence.',
            'sampling_time_benefit_seconds': None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=0)
    args = parser.parse_args()
    while True:
        value = audit(args.root)
        path = args.root / 'ik_exchange/observation.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(value, indent=2) + '\n')
        temp.replace(path)
        print(json.dumps({'observed_at': value['observed_at'], 'jobs': len(value['jobs']), 'errors': len(value['errors'])}), flush=True)
        if args.interval <= 0:
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
