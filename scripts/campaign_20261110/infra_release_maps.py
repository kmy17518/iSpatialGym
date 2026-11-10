"""Audit completed repair maps and build a new isolated combined data release."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def release(root, previous, destination, layers):
    from ispatialgym.sampling import attach_maps, reach_maps

    root, previous, destination = (Path(p).resolve() for p in (root, previous, destination))
    assert previous.is_relative_to(root) and destination.is_relative_to(root)
    assert not destination.exists(), 'release destination already exists'
    records = []
    for layer in layers:
        layer = Path(layer).resolve()
        assert layer.is_relative_to(root)
        paths = sorted((layer / 'maps').rglob('*.json'))
        assert paths, f'no maps in {layer}'
        for path in paths:
            module = attach_maps if path.name.startswith('amap__') else reach_maps
            prefix = path.name.startswith('prefix__')
            reader = module.AttachMap if module is attach_maps else (module.PrefixMap if prefix else module.ReachMap)
            record = reader(str(path))
            with np.load(path.with_suffix('.npz')) as arrays:
                status = [k for k in arrays.files if k.startswith('s')]
                if prefix:
                    assert record.meta['key'] == module.prefix_map_key(record.meta)
                    assert record.meta['arm'] in record.meta['arm_validated']
                    minimum = max(float(record.meta['min_open_rad']), 1e-4)
                    positive = sum(int((arrays[k] >= minimum-1e-6).sum()) for k in status)
                else:
                    positive = sum(int((arrays[k] == module.STATUS['feasible']).sum()) for k in status)
                assert status and positive > 0, f'map has no feasible cells: {path}'
                assert np.isfinite(arrays['cells']).all() and np.isfinite(arrays['yaws']).all()
                for name in status:
                    if prefix:
                        assert np.isfinite(arrays[name]).all()
                        assert arrays[name].max() <= float(record.meta['validated_travel_rad']) + 1e-5
                    else:
                        assert set(np.unique(arrays[name])).issubset(set(module.STATUS.values()))
            records.append({'source': str(path), 'relative_map': str(path.relative_to(layer/'maps')),
                            'positive': positive, 'sha256': digest(path),
                            'npz_sha256': digest(path.with_suffix('.npz'))})
    shutil.copytree(previous, destination, symlinks=False)
    for path in [destination, *destination.rglob('*')]:
        path.chmod(path.stat().st_mode | 0o200)
    for row in records:
        path = Path(row['source'])
        target = destination / 'maps' / row['relative_map']
        target.parent.mkdir(parents=True, exist_ok=True)
        for source, target_file in ((path, target), (path.with_suffix('.npz'), target.with_suffix('.npz'))):
            if target_file.exists():
                assert digest(source) == digest(target_file), f'conflicting map: {target_file}'
            else:
                shutil.copy2(source, target_file)
    (destination/'repair_content_audit.json').write_text(json.dumps({'maps': len(records), 'records': records}, indent=2)+'\n')
    manifest = json.loads((destination/'release_manifest.json').read_text())
    manifest.update(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), previous_release=str(previous),
                    new_repair_maps=manifest.get('new_repair_maps', 0)+len(records))
    manifest['files'] = {str(path.relative_to(destination)): digest(path) for path in sorted(destination.rglob('*'))
                         if path.is_file() and path.name != 'release_manifest.json'}
    (destination/'release_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'destination': str(destination), 'positive_maps': len(records), 'reader_check_pending': True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'previous', 'destination'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--layer', action='append', required=True)
    args = parser.parse_args()
    release(args.root, args.previous, args.destination, args.layer)
