"""Admit completed native contact modes independently, preserving exact evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_object_identity(obj, job):
    """Use the native library scale identity, retaining the input representation."""
    from ispatialgym.grasping.schema import canonical_scale

    for key in ('category', 'model'):
        assert obj[key] == job[key], f'contact {key} differs: {obj[key]!r} != {job[key]!r}'
    actual, required = canonical_scale(obj['scale']), canonical_scale(job['scale'])
    assert len(actual) == len(required) == 3, 'contact scale must have three axes'
    assert all(np.isfinite(actual)) and all(np.isfinite(required)), 'contact scale is not finite'
    assert actual == required, f'contact scale differs: {actual!r} != {required!r}'
    return dict(library_scale=obj['scale'], requested_scale=job['scale'],
                canonical_scale=list(actual), convention='native-schema-round-1e-4')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'profiles', 'contacts', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists() and a.output.resolve().is_relative_to(a.root.resolve())
    from ispatialgym.grasping import contact_library as CL
    from ispatialgym.grasping import contact_policy as CP
    from ispatialgym.sampling import contact_maps as CM
    from ispatialgym.sampling import reach_maps as RM
    from ispatialgym.sampling.v2 import tabletop as TT
    profiles = json.loads(a.profiles.read_text()); admitted, blocked, files = [], [], {}

    def add(path, destination):
        path = Path(path).resolve()
        assert path.is_file()
        row = dict(source=str(path), destination=destination, sha256=sha(path), bytes=path.stat().st_size)
        assert destination not in files or files[destination] == row
        files[destination] = row

    for request in profiles['requests']:
        job = request['job']; entry = job['id']; root = a.contacts / entry
        for kind in request['kinds']:
            row = dict(class_id=request['class_id'], split=request['split'], role=request['role'], entry=entry,
                       kind=kind, height=float(request['height_key']), asset_sha256=job['campaign_provenance']['asset_sha256'])
            try:
                family = 'brace' if 'brace' in kind else 'push'
                library = Path(CL.library_path(str(root), family, entry))
                validate = root / 'logs' / ('validate_0_' + family + '.log')
                map_log = root / 'logs' / ('maps_0_' + kind + '.log')
                assert validate.is_file() and 'GB_CONTACT_VALIDATE_DONE' in validate.read_text(), 'native contact validation not complete'
                assert map_log.is_file() and 'GB_CONTACT_MAPS_DONE' in map_log.read_text(), 'native contact map computation not complete'
                lib = CL.load_library(str(library)); obj = lib['object']
                assert lib['digest'] == CL._library_digest(lib), 'contact library identity changed'
                row['object_identity'] = verify_object_identity(obj, job)
                selected = CM.select_entries(lib, kind, ('valid',), arm=CM.KINDS[kind]['arm'])
                assert selected, 'zero physically validated contacts'
                pose_digest = CM.pose_set(kind, selected, ('valid',))['digest']
                mapped = CM.ContactMap.find(str(root / 'maps'), kind, entry, row['height'], rest_id=obj['rest_id'],
                                            pose_set_digest=pose_digest, knobs_require=TT.contact_map_knobs(kind),
                                            policy_key=CP.DEFAULT.key(CM.KINDS[kind]['arm']), robot=RM.robot_digest())
                assert mapped is not None, 'native map resolver rejected exact contact identity'
                meta = mapped.meta; path = Path(mapped.path)
                assert meta['key'] == CM.map_key(meta) and meta['library']['digest'] == lib['digest']
                assert abs(meta['height']['value'] - row['height']) < 1e-6 and CM.object_key(meta['object']) == CM.object_key(obj)
                assert np.isfinite(mapped.cells).all() and np.isfinite(mapped.yaws).all()
                status = [v for k, v in mapped.arrays.items() if k.startswith('s')]
                assert status and all(set(np.unique(v)) <= set(meta['status_codes'].values()) for v in status)
                positive = sum(int((v == meta['status_codes']['feasible']).sum()) for v in status)
                assert positive > 0, 'zero feasible native contact cells'
                log_rows = [json.loads(line) for line in (root / 'maps/logs' / ('maps_' + kind + '_0of1.jsonl')).read_text().splitlines() if line.strip()]
                proofs = [rec for rec in log_rows if rec.get('path') == str(path) and rec.get('ok') is True]
                assert len(proofs) == 1 and proofs[0]['n_feasible'] == positive, 'native map completion proof differs'
                add(path, 'maps/' + entry + '/' + path.name); add(path.with_suffix('.npz'), 'maps/' + entry + '/' + path.with_suffix('.npz').name)
                add(library, 'evidence/contact_libraries/' + entry + '/' + family + '/library.json')
                for name in ('library.json', 'geometry.json'):
                    add(Path(obj['source']) / name, 'evidence/contact_geometry/' + entry + '/' + name)
                add(validate, 'evidence/contact_logs/' + entry + '/' + validate.name)
                add(map_log, 'evidence/contact_logs/' + entry + '/' + map_log.name)
                row.update(status='native_contact_and_map_admitted', positive_contacts=len(selected), positive_cells=positive,
                           map=str(path), map_sha256=sha(path), library=str(library), library_sha256=sha(library),
                           pose_set_digest=pose_digest, rest_id=obj['rest_id'],
                           native_consumer_knobs=TT.contact_map_knobs(kind), native_required=kind in TT.CONTACT_MAP_REQUIRED,
                           process_shutdown_error_after_done='Fatal Python error' in map_log.read_text().split('GB_CONTACT_MAPS_DONE')[-1])
                admitted.append(row)
            except (AssertionError, OSError, ValueError, KeyError) as exc:
                row.update(status='blocked_or_incomplete', reason=str(exc)); blocked.append(row)
    add(a.profiles, 'evidence/contact_profiles.json')
    result = dict(schema='isg-native-contact-transfer-v1', source_profiles=str(a.profiles), source_profiles_sha256=sha(a.profiles),
                  active_policy_sha256=profiles['active_policy_sha256'], admitted=admitted, blocked=blocked,
                  files=list(files.values()), native_reader_errors=0, production_released=False,
                  task_geometry_demo_replay_still_required=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as stream:json.dump(result, stream, indent=2);stream.write('\n')
    print(json.dumps(dict(admitted_modes=len(admitted), blocked_modes=len(blocked), files=len(files), sha256=sha(a.output))))


if __name__ == '__main__':
    main()
