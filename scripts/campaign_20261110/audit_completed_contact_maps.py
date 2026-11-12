"""Finite additive audit; never edits original results, claims, or live verifier state."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from scripts.campaign_20261110 import reconcile_consumed_maps as reader


def audit(manifest, output):
    manifest, output = Path(manifest), Path(output)
    assert not output.exists()
    output.mkdir(parents=True)
    data = json.loads(manifest.read_text()); rows = []; cache = {}
    assert len(data['jobs']) <= 40
    for ticket in data['jobs']:
        assert ticket['computation_kind'] == 'contact_map' and ticket['consumer_kind'] == 'push_blade'
        directory = Path(ticket['prepared']['path']).parent; terminal = directory / 'terminal.json'
        row = {'physical_dedup_key': ticket['physical_dedup_key'], 'prepared': ticket['prepared'],
               'terminal': {'path': str(terminal), 'sha256': reader.sha(terminal)},
               'verification_started_at': time.time(), 'benchmark_credit': 0}
        try:
            row.update(reader.native_read_contact(ticket, terminal, directory / 'maps', 'cluster', cache,
                                                  allow_completed_teardown=True), verified_at=time.time())
        except (ValueError, OSError, KeyError, TypeError) as exc:
            row.update(status='failed', blocker=type(exc).__name__ + ': ' + str(exc), diagnosed_at=time.time())
        rows.append(row)
        reader.write(output / (ticket['physical_dedup_key'] + '.json'), row)
    result = {'schema': 'isg-completed-contact-artifact-audit-v1', 'observed_at': time.time(),
              'manifest': {'path': str(manifest), 'sha256': reader.sha(manifest)},
              'owner': '/root/templates_assets', 'rows': rows, 'summary': dict(Counter(r['status'] for r in rows)),
              'original_process_exits_preserved': True, 'live_verifier_modified': False,
              'benchmark_credit': 0, 'production_authorized': False}
    reader.write(output / 'audit.json', result)
    print(json.dumps({'audit': {'path': str(output / 'audit.json'), 'sha256': reader.sha(output / 'audit.json')},
                      'summary': result['summary']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--manifest', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); audit(a.manifest, a.output)
