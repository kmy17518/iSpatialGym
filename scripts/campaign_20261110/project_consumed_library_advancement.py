"""Finite scalar projection of verified-library downstream submission evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from scripts.campaign_20261110.contracts import write_json_atomic


def checked(ref):
    data = Path(ref['path']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == ref['sha256'], ref['path']
    return json.loads(data)


def project(root, binding_ref):
    root = Path(root); bindings = checked(binding_ref)
    assert bindings['schema'] == 'isg-library-downstream-bindings-v1'
    assert len(bindings['bindings']) <= 128
    results = []
    for binding in bindings['bindings']:
        proof = checked(binding['verification']); key = binding['library_physical_key']
        assert proof['physical_dedup_key'] == key and proof['status'] == 'verified_positive'
        path = root / 'infra/coordinator_dispatch/preparation_verification' / (key + '.json')
        side = json.loads(path.read_text())
        assert side['physical_dedup_key'] == key and side['status'] == 'verified_positive'
        assert side['verified_at'] == proof['verified_at']
        assert binding['verification'] in side['evidence']
        submitted, pending = {}, []
        for downstream in binding['downstream']:
            receipt = Path(downstream['submission_path'])
            assert receipt.is_relative_to(root / 'infra/coordination/preparation_dispatch')
            data = receipt.read_bytes() if receipt.exists() else None
            row = json.loads(data) if data is not None else {}
            at = row.get('submitted_at', row.get('observed_at'))
            physical = downstream['physical_dedup_key']
            if row.get('job_id') and row.get('physical_dedup_key') == physical and isinstance(at, (int, float)):
                submitted[physical] = {'submitted_at': at, 'job_id': row['job_id'],
                                       'receipt': {'path': str(receipt), 'sha256': hashlib.sha256(data).hexdigest()}}
            else:
                pending.append(physical)
        observed = time.time()
        side.update(downstream_required=True, downstream_ids=[d['physical_dedup_key'] for d in binding['downstream']],
                    downstream_submissions=submitted, downstream_observed_at=observed,
                    downstream_submitted_at=max((v['submitted_at'] for v in submitted.values()), default=None) if not pending else None,
                    blocker='Ready exact map tickets await root submission' if pending else None,
                    next_check_at=observed + 60 if pending else None, downstream_binding=binding_ref)
        assert binding['downstream'], 'Positive library has no enumerated map consumer'
        write_json_atomic(path, side)
        results.append({'physical_dedup_key': key, 'submitted': len(submitted), 'pending': len(pending)})
    return {'observed_at': time.time(), 'rows': results, 'dispatch_authority': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'bindings', 'bindings-sha256'): p.add_argument('--' + name, required=True)
    a = p.parse_args()
    print(json.dumps(project(a.root, {'path': a.bindings, 'sha256': a.bindings_sha256})))
