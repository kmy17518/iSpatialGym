"""Read actual selected grasp results through the native resolver.

A successful process or a library file with no tested grasps is not a positive
dependency. This audit does not authorize task generation or acceptance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.campaign_20261110.contracts import file_digest, require, write_json_atomic
from scripts.campaign_20261110.reservation_policy import active_policy


def reference(path):
    return dict(path=str(Path(path).resolve()), sha256=file_digest(path))


def validate_record(job, record, reservations):
    key = job['category'] + '/' + job['model']
    require(job.get('ready') is True and job.get('readiness_source') and job.get('asset_sha256'),
            'ready asset provenance missing')
    require(key not in reservations['train_exclude_models'] and
            job['category'] not in reservations['category_holdout_categories'], 'reserved training asset')
    require(record.get('schema') == 'isg-grasp-library/2', 'unvalidated library schema')
    require(all(record.get(k) == job[k] for k in ('category', 'model', 'dataset', 'scale', 'constraint')),
            'exact model, dataset, scale or constraint mismatch')
    floating = job['constraint'] == 'floating'
    validator = 'isg-floating-validator/2' if floating else 'isg-tabletop-validator/1'
    require(record.get('provenance', {}).get('validator') == validator, 'native physics validator mismatch')
    require(record.get('physics_profile') and record.get('policy_key'), 'physics/contact identity missing')
    grasps = record.get('grasps')
    require(grasps and all(g.get('physics') == 'tested' for g in grasps), 'zero positive tested grasps')
    if floating:
        require(not record.get('rests') and all(g.get('rest_id') is None for g in grasps),
                'floating library contains a support rest')
        tasks = job.get('held_tasks', [])
        require(tasks, 'floating task motion requirements missing')
        for grasp in grasps:
            require(all(grasp.get('held_motions', {}).get(task, {}).get('motions_executed', 0) > 0
                        for task in tasks), 'required held-object motion was not executed')
    else:
        require(all(g.get('rest_id') in record.get('rests', {}) for g in grasps),
                'positive grasp lacks its measured rest')
    return len(grasps)


def audit(root, inputs):
    from ispatialgym.grasping import schema as S
    from ispatialgym.sampling.knowledge import is_ready

    pointer, reservations, _ = active_policy(root)
    require(pointer is not None, 'current reservation policy missing')
    verified, failed, missing = [], [], []
    for source in inputs:
        manifest = Path(source['manifest']['path'])
        require(file_digest(manifest) == source['manifest']['sha256'], 'selected job manifest changed')
        jobs = json.loads(manifest.read_text())
        for index in source['indices']:
            job = jobs[index]
            library_root = Path(source['library_root'])
            path = library_root/'models'/job['id']/'library.json'
            row = dict(entry_id=job['id'], source_manifest=source['manifest'], source_index=index,
                       job=job, source_job=source.get('source_job'), library_root=str(library_root))
            if not path.exists():
                missing.append({**row, 'status': 'unavailable', 'reason': 'no native library output observed'})
                continue
            try:
                data = json.loads(path.read_text())
                count = validate_record(job, data, reservations)
                require(file_digest(job['asset_path']) == job['asset_sha256'], 'asset bytes changed')
                require(is_ready(job['model']), 'current native readiness failed')
                lib = S.load_model(job['category'], job['model'], job['scale'], str(library_root),
                                   constraint=job['constraint'], rest_hint_wxyz=job.get('rest_hint_wxyz'))
                require(lib is not None and Path(lib.path).resolve() == path.resolve(), 'native resolver identity differs')
                S.check_compatible(lib, scale=job['scale'])
                require(len(lib.grasps()) == count, 'native reader positive count differs')
                for rest in {g['rest_id'] for g in data['grasps'] if g.get('rest_id') is not None}:
                    S.check_compatible(lib, rest_id=rest, rest_kind='stable')
                closure = [reference(p) for p in sorted(path.parent.iterdir()) if p.is_file()]
                verified.append({**row, 'status': 'verified', 'library': reference(path),
                    'native_reader_passed': True, 'native_identity': lib.identity(),
                    'native_identity_key': lib.identity_key(), 'positive_grasps': count,
                    'physics_profile': data['physics_profile'], 'policy_key': data['policy_key'],
                    'rest_ids': sorted(lib.rests), 'files': closure})
            except OSError as exc:
                # Shared-account atomic files can exist without being readable.
                # An observation failure is not evidence of failed physics, and
                # trying to hash the same unreadable file would abort the batch.
                missing.append({**row, 'status': 'unavailable',
                    'library': {'path': str(path.resolve()), 'sha256': None},
                    'reason': str(exc), 'error_type': type(exc).__name__})
            except (ValueError, KeyError, TypeError) as exc:
                try:
                    library_ref = reference(path)
                except OSError as observation_error:
                    missing.append({**row, 'status': 'unavailable',
                        'library': {'path': str(path.resolve()), 'sha256': None},
                        'reason': str(observation_error),
                        'error_type': type(observation_error).__name__,
                        'unconfirmed_validation_error': str(exc)})
                else:
                    failed.append({**row, 'status': 'failed', 'library': library_ref,
                                   'reason': str(exc), 'error_type': type(exc).__name__})
    return dict(schema='isg-native-library-harvest-v1', campaign_id='campaign_20261110',
        observed_at=datetime.now(timezone.utc).isoformat(), owner='/root/templates_assets',
        reservation_policy=pointer, inputs=inputs, native_reader=reference(S.__file__),
        verified=verified, failed=failed, unavailable=missing, production_authorization=False,
        summary=dict(attempted_entries=len(verified)+len(failed)+len(missing), verified=len(verified),
                     failed=len(failed), unavailable=len(missing)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','inputs','output'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    require(not a.output.exists() and a.output.resolve().is_relative_to(a.root.resolve()), 'fresh owned audit required')
    result=audit(a.root,json.loads(a.inputs.read_text())['inputs'])
    result['inputs_manifest']=reference(a.inputs)
    write_json_atomic(a.output,result)
    print(json.dumps({**result['summary'], 'audit':reference(a.output)}))


if __name__=='__main__': main()
