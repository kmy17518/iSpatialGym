"""Prepare/adopt additive forensic proofs through the existing verifier's recovery path.

Root executes --install after review. No state, terminal, claim, or receipt edits;
the sole live reader adopts immutable proofs on its next ordinary scan.
"""
import argparse
import json
import os
from pathlib import Path
import time
from scripts.campaign_20261110.reconcile_consumed_maps import checked, checked_hash, require, sha


def install(audit_ref, output, execute=False):
    cache = {}; audit = checked(audit_ref, cache); output = Path(output)
    require(audit['schema'] == 'isg-completed-contact-artifact-audit-v1', 'Unsupported forensic audit')
    manifest = checked(audit['manifest'], cache)
    tickets = {t['physical_dedup_key']: t for t in manifest['jobs']}
    state = json.loads((output / 'state.json').read_text()); planned = []
    for row in audit['rows']:
        if row['status'] not in {'verified_positive', 'valid_zero'}: continue
        require(row['native_reader_passed'] and row['process_exit_success'] is False and row['original_returncode'] == -11,
                'Forensic proof disposition differs')
        key = row['physical_dedup_key']; ticket = tickets[key]; prepared = checked(ticket['prepared'], cache)
        require(prepared['physical_dedup_key'] == key and row['prepared'] == ticket['prepared'], 'Forensic preparation differs')
        for ref in ticket['input_refs'] + [ticket['runtime_ref'], row['terminal']] + row['verified_files'] + row['completion_evidence']:
            checked_hash(ref, cache)
        old = state['rows'][key]
        require(old['terminal'] == row['terminal'], 'Current terminal differs')
        path = output / 'verifications' / key / (row['terminal']['sha256'] + '.json')
        proof = {**old, **row, 'verification_at': row['verified_at'], 'blocker': None,
                 'forensic_audit': audit_ref, 'adoption_method': 'existing immutable-proof recovery path'}
        if path.exists():
            retained = json.loads(path.read_text())
            require(retained.get('forensic_audit') == audit_ref and retained['terminal'] == row['terminal'], 'Competing proof exists')
        else:
            require(old['status'] == 'invalid_or_failed' and old.get('verification_at') is None, 'Source was not an unverified failure')
            if execute:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(path.name + '.forensic-' + str(os.getpid()))
                with temporary.open('x') as stream:
                    stream.write(json.dumps(proof, indent=2) + '\n'); stream.flush(); os.fsync(stream.fileno())
                try: os.link(temporary, path)  # no overwrite; native reader sees only complete JSON
                finally: temporary.unlink()
        planned.append({'physical_dedup_key': key, 'proof_path': str(path)})
    return {'observed_at': time.time(), 'audit': audit_ref, 'installed': execute,
            'proof_count': len(planned), 'proofs': planned, 'state_modified': False,
            'claims_modified': False, 'dispatch': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', required=True); p.add_argument('--sha256', required=True)
    p.add_argument('--output', required=True); p.add_argument('--install', action='store_true')
    a = p.parse_args(); print(json.dumps(install({'path': a.audit, 'sha256': a.sha256}, a.output, a.install)))
