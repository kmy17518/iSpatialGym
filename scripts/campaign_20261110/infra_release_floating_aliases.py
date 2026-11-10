"""Add native held-task routing aliases without changing validated grasp data."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physical_digest(data):
    return hashlib.sha256(json.dumps({k: v for k, v in data.items() if k != 'legacy_keys'},
                                    sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def replace_json(path, data):
    temporary = path.with_suffix(path.suffix + '.new')
    temporary.write_text(json.dumps(data, sort_keys=True, indent=2) + '\n')
    temporary.replace(path)  # Detach any hard link before changing owned metadata.


def release(root, previous, destination):
    from ispatialgym.grasping import schema as S
    from ispatialgym.sampling import grasp_library as G
    from ispatialgym.sampling.v2.tasks import held_object as H
    root, previous, destination = map(lambda p: Path(p).resolve(), (root, previous, destination))
    assert previous.is_relative_to(root) and destination.is_relative_to(root) and not destination.exists()
    manifest = json.loads((previous / 'release_manifest.json').read_text())
    assert all(sha(previous / name) == digest for name, digest in manifest['files'].items()), 'prior release changed'
    original = json.loads((previous / 'floating_content_audit.json').read_text())['records']
    shutil.copytree(previous, destination, copy_function=os.link, symlinks=False)
    for path in [destination, *destination.rglob('*')]:
        if path.is_dir(): path.chmod(path.stat().st_mode | 0o200)
    block = destination / 'release_block.json'
    block.write_text('{"reason":"native held floating route audit pending"}\n')
    os.environ['ISPATIALGYM_GRASP_LIBRARY_V2'] = str(destination / 'glib')
    os.environ['ISPATIALGYM_V2_ONLY'] = '1'
    G._v2_index.cache_clear()
    index = G._v2_index(str(destination / 'glib'))
    records = []
    for row in original:
        path = destination / 'glib/models' / row['id'] / 'library.json'
        before = json.loads(path.read_text())
        assert sha(path) == row['library_sha256'], 'physical source evidence differs'
        stem = before['category'] + '__' + before['model']
        old = row['alias']
        if old.startswith(stem) and old[len(stem):] in H.FLOATING_SUFFIXES:
            alias = old
        else:
            candidates = [stem + suffix for suffix in ('__f', '__sf', '__h', '')]
            alias = next((key for key in candidates if key not in index or Path(index[key]).resolve() == path), None)
            assert alias is not None, 'all native floating aliases already bind different libraries: ' + stem
            updated = {**before, 'legacy_keys': list(dict.fromkeys([*before.get('legacy_keys', []), alias]))}
            assert physical_digest(before) == physical_digest(updated), 'routing edit changes physical data'
            replace_json(path, updated)
            index[alias] = str(path)
        G._v2_index.cache_clear()
        assert H.floating_key(before['category'], before['model'], alias) == alias
        H.check_v2_floating(alias, row['scale'])
        lib = G.v2_entry(alias)
        assert lib is not None and Path(lib.path).resolve() == path and lib.floating
        S.check_compatible(lib, scale=row['scale'])
        G._check_v2_policy(lib, lib.data)
        assert len(G.grasps_from_v2(lib, alias)) > 0 and len(lib.grasps()) == row['positive_grasps']
        records.append({**row, 'old_alias': old, 'alias': alias, 'library': str(path),
                        'source_library_sha256': row['library_sha256'], 'library_sha256': sha(path),
                        'physical_payload_sha256': physical_digest(before)})
    audit = {'schema': 'isg-native-floating-route-release-v1', 'previous_release': str(previous),
             'records': records, 'native_task_reader_errors': 0,
             'physical_grasps_rests_geometry_unchanged': True, 'task_demo_replay_required': True}
    replace_json(destination / 'floating_alias_audit.json', audit)
    block.unlink()
    manifest.update(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), previous_release=str(previous),
                    native_floating_routes=len(records))
    manifest['files'] = {str(p.relative_to(destination)): sha(p) for p in sorted(destination.rglob('*'))
                         if p.is_file() and p.name != 'release_manifest.json'}
    replace_json(destination / 'release_manifest.json', manifest)
    assert all(sha(previous / name) == digest for name, digest in json.loads((previous / 'release_manifest.json').read_text())['files'].items()), 'source bytes changed'
    for path in [destination, *destination.rglob('*')]:
        if path.stat().st_mode & 0o222: path.chmod(path.stat().st_mode & ~0o222)
    print(json.dumps({'release': str(destination), 'native_routes': len(records), 'audit': str(destination / 'floating_alias_audit.json')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'previous', 'destination'): parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    release(args.root, args.previous, args.destination)
