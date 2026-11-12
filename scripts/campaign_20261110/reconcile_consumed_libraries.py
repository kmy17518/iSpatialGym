"""Finite or warmed CPU library verification; no submit or provider operations."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

from scripts.campaign_20261110.contracts import write_json_atomic


def ref(path):
    path = Path(path)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def cycle(root, manifest, expected_sha, output, *, budget=15, auditor=None):
    if auditor is None:
        from scripts.campaign_20261110.audit_consumed_attachment_libraries import audit
        auditor = audit
    root, manifest, output = map(Path, (root, manifest, output)); output.mkdir(parents=True, exist_ok=True)
    assert ref(manifest)['sha256'] == expected_sha
    lock = output / 'scan.lock'
    try: lock.mkdir()
    except FileExistsError: return {'status': 'existing_verifier_lock', 'observed_at': time.time()}
    started = time.monotonic()
    try:
        tickets = json.loads(manifest.read_text())['jobs']; state_path = output / 'state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {'rows': {}, 'cursor': 0}
        new = 0; inspected = 0; cursor = state['cursor'] % max(1, len(tickets))
        while inspected < len(tickets):
            ticket = tickets[(cursor + inspected) % len(tickets)]; inspected += 1
            key = ticket['physical_dedup_key']; terminal = Path(ticket['prepared']['path']).parent / 'terminal.json'
            if not terminal.exists(): continue
            terminal_ref = ref(terminal); old = state['rows'].get(key)
            if old and old.get('terminal') == terminal_ref: continue
            if new and time.monotonic() - started >= budget: break
            verification_started = time.time(); result = auditor(root, manifest, only_keys={key})
            assert len(result['rows']) == 1
            row = result['rows'][0]; completed = time.time(); terminal_doc = json.loads(terminal.read_text())
            status = {'verified_positive': 'verified_positive', 'valid_zero': 'verified_zero'}.get(row['status'], 'failed')
            proof = output / 'verifications' / key / (terminal_ref['sha256'] + '.json')
            value = {**row, 'status': status, 'terminal': terminal_ref, 'terminal_at': terminal_doc.get('finished_at'),
                     'verification_started_at': verification_started, 'verified_at': completed,
                     'source_manifest': ref(manifest), 'native_reader': result['native_reader'],
                     'native_candidate_generator': result['native_candidate_generator']}
            if proof.exists():
                value = json.loads(proof.read_text())
            else: write_json_atomic(proof, value)
            positive = status == 'verified_positive'
            sidecar = {'schema': 'isg-consumed-library-verification-v1', 'physical_dedup_key': key,
                       'owner': '/root/templates_assets', 'terminal_at': value['terminal_at'],
                       'verification_started_at': value['verification_started_at'], 'verified_at': value['verified_at'],
                       'status': status, 'job_id': row.get('job_id'), 'evidence': [ref(proof), terminal_ref],
                       'downstream_required': positive, 'downstream_ids': [], 'downstream_submitted_at': None,
                       'blocker': 'Exact per-height specialized map and alias closure pending' if positive else row.get('reason'),
                       'next_check_at': completed + 60 if positive else None,
                       'benchmark_credit': 0, 'production_authorized': False}
            write_json_atomic(root / 'infra/coordinator_dispatch/preparation_verification' / (key + '.json'), sidecar)
            state['rows'][key] = {**value, 'verification': ref(proof)}; new += 1
            write_json_atomic(state_path, state)
        state['cursor'] = (cursor + inspected) % max(1, len(tickets)); write_json_atomic(state_path, state)
        counts = Counter(r['status'] for r in state['rows'].values())
        counts['awaiting_terminal'] = len(tickets) - len(state['rows'])
        status = {'schema': 'isg-consumed-library-verifier-v1', 'observed_at': time.time(), 'owner': '/root/templates_assets',
                  'source_manifest': ref(manifest), 'summary': dict(counts), 'verified_this_cycle': new,
                  'cycle_seconds': time.monotonic() - started, 'production_authorized': False, 'benchmark_credit': 0}
        write_json_atomic(output / 'status.json', status); return status
    finally: lock.rmdir()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'manifest', 'manifest-sha256', 'output'): p.add_argument('--' + name, required=True)
    p.add_argument('--watch', action='store_true'); p.add_argument('--interval', type=float, default=30)
    p.add_argument('--max-cycles', type=int, default=720); p.add_argument('--max-cycle-seconds', type=float, default=15)
    a = p.parse_args(); assert 15 <= a.interval <= 60 and 1 <= a.max_cycles <= 720 and 1 <= a.max_cycle_seconds <= 20
    output = Path(a.output); output.mkdir(parents=True, exist_ok=True); watch_lock = output / 'watch.lock'
    if a.watch: watch_lock.mkdir()
    try:
        t = time.time(); startup = {'started_at': t, 'phase': 'importing_native_reader', 'job_id': os.environ.get('SLURM_JOB_ID')}
        write_json_atomic(output / 'worker_startup.json', startup)
        from ispatialgym.grasping import schema, candidates
        write_json_atomic(output / 'worker_startup.json', {**startup, 'phase': 'native_reader_ready', 'finished_at': time.time(), 'cold_start_seconds': time.time() - t})
        for _ in range(a.max_cycles if a.watch else 1):
            if (output / 'STOP_AFTER_CYCLE').exists(): break
            started = time.monotonic()
            print(json.dumps(cycle(a.root, a.manifest, a.manifest_sha256, a.output, budget=a.max_cycle_seconds)), flush=True)
            if a.watch: time.sleep(max(0, a.interval - (time.monotonic() - started)))
    finally:
        if a.watch: watch_lock.rmdir()


if __name__ == '__main__': main()
