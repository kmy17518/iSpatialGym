"""Publish exact attachment maps with their native library/spec/hover identities."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def merge(root, previous, admission, destination):
    from ispatialgym.grasping import schema as S, contact_policy as CP
    from ispatialgym.sampling import attach_maps as AM, grasp_library as G, reach_maps as RM
    from ispatialgym.sampling.generation_defaults import defaults_for

    root, previous, admission, destination = map(lambda p: Path(p).resolve(), (root, previous, admission, destination))
    assert all(p.is_relative_to(root) for p in (previous, admission, destination)) and not destination.exists()
    transfer = json.loads(admission.read_text())
    assert transfer['schema'] == 'campaign-attachment-transfer-v1' and transfer['admitted']
    assert transfer['native_reader_errors'] == 0
    token = sha(admission)[:16]
    before = json.loads((previous / 'release_manifest.json').read_text())
    os.umask(0o027); destination.mkdir(mode=0o750)
    (destination / 'release_block.json').write_text('{"reason":"attachment reader verification pending"}\n')
    copied = {}; sources = {}

    def copy(path, relative, expected):
        path = Path(path).resolve(); target = destination / relative
        assert path.is_relative_to(root) and target.resolve().is_relative_to(destination)
        data = path.read_bytes(); assert hashlib.sha256(data).hexdigest() == expected
        if target.exists():
            assert sha(target) == expected, 'conflicting release path: ' + relative
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.open('xb').write(data)
        copied[relative] = expected
        return target

    for relative, expected in before['files'].items():
        copy(previous / relative, relative, expected)
    for row in transfer['files']:
        relative = row['destination']
        if relative.startswith('evidence/'):
            relative = 'evidence/' + token + '/' + relative.removeprefix('evidence/')
        sources[row['source']] = copy(row['source'], relative, row['sha256'])
    transfer_name = 'attachment_transfer_audit_' + token + '.json'
    copy(admission, transfer_name, sha(admission))
    checked = []
    for row in transfer['admitted']:
        lp = destination / 'glib/models' / row['entry'] / 'library.json'
        assert sha(lp) == row['library_sha256']
        ld = json.loads(lp.read_text())
        lib = S.load_model(ld['category'], ld['model'], ld['scale'], str(destination / 'glib'),
                           constraint=ld['constraint'], rest_hint_wxyz=ld.get('rest_hint_wxyz'))
        assert lib and Path(lib.path).resolve() == lp.resolve() and lib.grasps()
        spec_path = sources[row['spec_path']]; assert sha(spec_path) == row['spec_sha256']
        spec = AM.load_spec(str(spec_path)); assert spec.digest() == row['spec_digest']
        partner = row['partner']
        assert all(spec.data['partner'][key] == partner[key] for key in ('category', 'model', 'scale'))
        for record in row['maps']:
            mapped = AM.AttachMap.find(str(destination / 'maps' / row['entry']), .7588,
                partner=(partner['category'], partner['model'], partner['scale']), rest_id=G.v2_rest_id(lib),
                object_key=lib.identity_key(), arm='right', hover=AM.oracle_hover_params(record['rule']),
                pre_back_m=defaults_for('configuration_matching-attachment')['pre_attach_back_m'],
                policy_key=CP.DEFAULT.key('right'), robot=RM.robot_digest())
            assert mapped and sha(mapped.path) == record['sha256']
            assert mapped.meta['spec']['digest'] == spec.digest() and mapped.meta['male']['entry'] == row['entry']
            assert np.isfinite(mapped.cells).all() and np.isfinite(mapped.yaws).all()
            values = [v for k, v in mapped.arrays.items() if k.startswith('s')]
            assert values and all(set(np.unique(v)) <= set(AM.STATUS.values()) for v in values)
            positive = sum(int((v == AM.STATUS['feasible']).sum()) for v in values)
            assert positive == record['positive_cells'] and positive > 0
            checked.append({'class_id': row['class_id'], 'entry': row['entry'], 'rule': record['rule'],
                            'map': mapped.path, 'positive_cells': positive})
    audit_name = 'attachment_reader_audit_' + token + '.json'
    audit = {'native_reader_errors': 0, 'checked': checked, 'source_permissions_unchanged': True,
             'task_geometry_demo_replay_still_required': True}
    (destination / audit_name).write_text(json.dumps(audit, indent=2) + '\n')
    copied[audit_name] = sha(destination / audit_name)
    manifest = {**before, 'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'previous_release': str(previous), 'new_attachment_maps': len(checked),
                'new_attachment_reader_audit': audit_name, 'new_attachment_transfer_audit': transfer_name, 'files': copied}
    (destination / 'release_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (destination / 'release_block.json').unlink()
    print(json.dumps({'release': str(destination), 'positive_attachment_maps': len(checked),
                      'manifest_sha256': sha(destination / 'release_manifest.json')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'previous', 'admission', 'destination'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    merge(a.root, a.previous, a.admission, a.destination)
