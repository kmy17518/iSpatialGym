"""Copy admitted contact maps into an isolated release and recheck native readers."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def merge(root, previous, inputs, destination):
    from ispatialgym.grasping import contact_library as CL, contact_policy as CP
    from ispatialgym.sampling import contact_maps as CM, reach_maps as RM
    from ispatialgym.sampling.v2 import tabletop as TT
    from ispatialgym.sampling.height_anchors import AnchorSet

    root, previous, inputs, destination = map(lambda p: Path(p).resolve(), (root, previous, inputs, destination))
    assert all(p.is_relative_to(root) for p in (previous, inputs, destination)) and not destination.exists()
    transfer = json.loads((inputs / 'transfer_manifest.json').read_text())
    assert transfer['schema'] == 'isg-native-contact-transfer-v1' and transfer['admitted']
    assert transfer['native_reader_errors'] == 0
    token = sha(inputs / 'transfer_manifest.json')[:16]
    before = json.loads((previous / 'release_manifest.json').read_text())
    os.umask(0o027)
    destination.mkdir(mode=0o750)
    (destination / 'release_block.json').write_text('{"reason":"contact reader verification pending"}\n')
    copied = {}

    def copy(path, relative, expected):
        target = destination / relative
        assert target.resolve().is_relative_to(destination)
        data = Path(path).read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected, 'source bytes changed: ' + str(path)
        if target.exists():
            assert sha(target) == expected, 'conflicting release path: ' + relative
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.open('xb').write(data)
        copied[relative] = expected

    for relative, expected in before['files'].items():
        copy(previous / relative, relative, expected)
    by_source = {}
    for row in transfer['files']:
        relative = row['destination']
        if relative.startswith('evidence/'):
            relative = 'evidence/' + token + '/' + relative.removeprefix('evidence/')
        copy(inputs / row['destination'], relative, row['sha256'])
        by_source[row['source']] = destination / relative
    transfer_name = 'contact_transfer_audit_' + token + '.json'
    copy(inputs / 'transfer_manifest.json', transfer_name, sha(inputs / 'transfer_manifest.json'))
    anchors_raw = json.loads((destination / 'anchors.json').read_text())
    anchors = AnchorSet(anchors_raw['anchors'], anchors_raw['max_approx_m'])
    checked = []
    for row in transfer['admitted']:
        library = by_source[row['library']]
        assert sha(library) == row['library_sha256']
        lib = CL.load_library(str(library))
        assert lib['digest'] == CL._library_digest(lib)
        kind = row['kind']; arm = CM.KINDS[kind]['arm']
        entries = CM.select_entries(lib, kind, ('valid',), arm=arm)
        assert len(entries) == row['positive_contacts'] and entries
        pose_digest = CM.pose_set(kind, entries, ('valid',))['digest']
        assert pose_digest == row['pose_set_digest']
        anchor, how = anchors.lookup(row['height'])
        assert anchor and how == 'exact' and abs(anchor['height'] - row['height']) < 1e-8
        native = CM.ContactMap.find(str(destination / 'maps'), kind, row['entry'], row['height'],
                                   rest_id=row['rest_id'], arm=arm, anchors=anchors,
                                   pose_set_digest=pose_digest, knobs_require=TT.contact_map_knobs(kind),
                                   policy_key=CP.DEFAULT.key(arm), robot=RM.robot_digest())
        assert native and sha(native.path) == row['map_sha256']
        assert native.meta['key'] == CM.map_key(native.meta)
        assert native.meta['library']['digest'] == lib['digest']
        assert CM.object_key(native.meta['object']) == CM.object_key(lib['object'])
        assert np.isfinite(native.cells).all() and np.isfinite(native.yaws).all()
        arrays = [values for name, values in native.arrays.items() if name.startswith('s')]
        codes = native.meta['status_codes']
        assert arrays and all(set(np.unique(values)) <= set(codes.values()) for values in arrays)
        positive = sum(int((values == codes['feasible']).sum()) for values in arrays)
        assert positive == row['positive_cells'] and positive > 0
        checked.append({**row, 'released_map': native.path, 'released_library': str(library)})
    audit_name = 'contact_reader_audit_' + token + '.json'
    audit = {'native_reader_errors': 0, 'positive_contact_maps': len(checked), 'checked': checked,
             'source_permissions_unchanged': True, 'task_geometry_demo_replay_still_required': True}
    (destination / audit_name).write_text(json.dumps(audit, indent=2) + '\n')
    copied[audit_name] = sha(destination / audit_name)
    manifest = {**before, 'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'previous_release': str(previous), 'new_contact_maps': len(checked),
                'new_contact_reader_audit': audit_name, 'new_contact_transfer_audit': transfer_name,
                'files': copied}
    (destination / 'release_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (destination / 'release_block.json').unlink()
    print(json.dumps({'release': str(destination), 'positive_contact_maps': len(checked),
                      'manifest_sha256': sha(destination / 'release_manifest.json')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'previous', 'inputs', 'destination'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    merge(a.root, a.previous, a.inputs, a.destination)
