"""Publish completed generic maps after exact native resolver and status checks."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def merge(root, previous, inputs, destination, fingers_q=None):
    from ispatialgym.grasping import contact_policy as CP
    from ispatialgym.sampling import reach_maps as R
    from ispatialgym.sampling.height_anchors import AnchorSet

    root, previous, inputs, destination = map(lambda x: Path(x).resolve(), (root, previous, inputs, destination))
    assert all(p.is_relative_to(root) for p in (previous, inputs, destination)) and not destination.exists()
    logs = sorted((inputs / 'logs').glob('maps_*_generic.log'))
    assert len(logs) == 2 and all('GB_MAP_DONE' in p.read_text() and 'Simulation App Shutting Down' in p.read_text() for p in logs)
    os.umask(0o027)
    destination.mkdir(mode=0o750)
    (destination / 'release_block.json').write_text('{"reason":"generic native reader checks pending"}\n')
    before = json.loads((previous / 'release_manifest.json').read_text())
    files = {}

    def copy(source, relative, expected):
        source = Path(source).resolve()
        target = (destination / relative).resolve()
        assert source.is_relative_to(root) and target.is_relative_to(destination)
        content = source.read_bytes()
        assert hashlib.sha256(content).hexdigest() == expected
        if target.exists():
            assert sha(target) == expected, 'conflicting dependency bytes'
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.open('xb').write(content)
        files[relative] = expected

    for relative, expected in before['files'].items():
        copy(previous / relative, relative, expected)
    anchor_data = json.loads((destination / 'anchors.json').read_text())
    anchors = AnchorSet(anchor_data['anchors'], anchor_data['max_approx_m'])
    checked = []
    coverage = {}
    for source in sorted((inputs / 'maps' / R.GENERIC_DIR).glob('*.json')):
        native = R.ReachMap(str(source))
        meta = native.meta
        assert meta['kind'] == R.GENERIC_KIND and meta['key'] == R.generic_map_key(meta)
        arm, height = meta['arm'], float(meta['height']['value'])
        assert arm in ('right', 'left') and meta['pose_set'] == R.generic_pose_set(.03, fingers_q)
        assert meta['policy_key'] == CP.DEFAULT.key(arm) and meta['robot'] == R.robot_digest()
        assert np.isfinite(native.cells).all() and np.isfinite(native.yaws).all()
        states = [values for key, values in native.arrays.items() if key.startswith('s')]
        assert states and all(set(np.unique(v)) <= set(R.STATUS.values()) for v in states)
        positive = sum(int((v == R.STATUS['feasible']).sum()) for v in states)
        assert positive > 0
        found = R.find_generic(str(source.parent), height, arm=arm, pose_set=meta['pose_set'],
                               policy_key=CP.DEFAULT.key(arm), robot=R.robot_digest(), anchors=anchors)
        assert found and Path(found.path).resolve() == source and found.lookup['how'] == 'exact'
        relative = 'maps/' + R.GENERIC_DIR + '/' + source.name
        copy(source, relative, sha(source))
        copy(source.with_suffix('.npz'), relative[:-5] + '.npz', sha(source.with_suffix('.npz')))
        after = R.find_generic(str(destination / 'maps' / R.GENERIC_DIR), height, arm=arm, pose_set=meta['pose_set'],
                              policy_key=CP.DEFAULT.key(arm), robot=R.robot_digest(), anchors=anchors)
        assert after and sha(after.path) == sha(source) and after.lookup['how'] == 'exact'
        coverage.setdefault(height, set()).add(arm)
        checked.append(dict(source=str(source), source_sha256=sha(source), path=after.path,
                            height=height, arm=arm, positive_cells=positive))
    assert checked and all(arms == {'left', 'right'} for arms in coverage.values())
    for log in logs:
        copy(log, 'evidence/generic_' + sha(log)[:16] + '_' + log.name, sha(log))
    audit = destination / 'generic_reader_audit.json'
    assert not audit.exists()
    audit.write_text(json.dumps(dict(native_reader_errors=0, checked=checked, source_permissions_unchanged=True,
                                    task_geometry_demo_replay_still_required=True), indent=2) + '\n')
    files[audit.name] = sha(audit)
    manifest = {**before, 'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'previous_release': str(previous), 'new_generic_maps': len(checked),
                'new_generic_reader_audit': audit.name, 'files': files}
    (destination / 'release_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (destination / 'release_block.json').unlink()
    print(json.dumps(dict(release=str(destination), native_generic_maps=len(checked),
                          manifest_sha256=sha(destination / 'release_manifest.json'))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'previous', 'inputs', 'destination'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--fingers-q', type=float, choices=(.05,), default=None,
                   help='Omit to require the production default finger-opening identity')
    a = p.parse_args()
    merge(a.root, a.previous, a.inputs, a.destination, a.fingers_q)
