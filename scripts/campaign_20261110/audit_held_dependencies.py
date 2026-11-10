#!/usr/bin/env python3
"""Admit completed exact held-target libraries for independent dependency maps."""
import argparse
import json
import os
from pathlib import Path

from scripts.campaign_20261110.contracts import file_digest, require, write_json_atomic
from scripts.campaign_20261110.reservation_policy import policy_digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'secondary-root', 'reservations', 'asset-hashes', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args(); require(not a.out.exists() and a.out.resolve().is_relative_to(a.secondary_root.resolve()), 'new owned output required')
    policy, hashes = json.loads(a.reservations.read_text()), json.loads(a.asset_hashes.read_text())
    require(hashes['reservation_sha256'] == file_digest(a.reservations), 'policy companion differs')
    from ispatialgym.grasping import schema as S
    from ispatialgym.sampling import grasp_library as G
    native_trays = a.root / 'heldout_probes/lef_tray_floors_v1/tray_floor_probe.json'
    trays = json.loads(native_trays.read_text()); require(trays['status'] == 'native_tray_floor_measurement_passed', 'native tray proof incomplete')
    batches = [('attachment', a.secondary_root / 'heldattachments_prebuild_v1', a.secondary_root / 'manifests/heldattachments_v1.json', [trays['tabletop_height_key']]),
               ('disembedding', a.secondary_root / 'helddisembedding_prebuild_v1', a.secondary_root / 'manifests/held_disembedding_v1.json',
                [trays['objects'][model]['map_height_key'] for model in ('gsxbym', 'mhhoga')])]
    a.out.mkdir(parents=True); admitted, skipped, outputs = [], [], []
    for kind, source, manifest, heights in batches:
        source_jobs = json.loads(manifest.read_text()); kept = []
        os.environ['ISPATIALGYM_GRASP_LIBRARY_V2'] = str(source / 'glib'); G._v2_index.cache_clear()
        for index, job in enumerate(source_jobs):
            base = dict(kind=kind, source_index=index, id=job['id'], source_manifest=str(manifest), source_manifest_sha256=file_digest(manifest), campaign_classes=job['campaign_classes'])
            try:
                key = job['category'] + '/' + job['model']; require(job['ready'] and job['readiness_source'], 'unready source model')
                require(key in policy['model_holdout_models'], 'model no longer reserved')
                expected = hashes['ready_assets'][key]
                require(expected['ready'] and job['asset_sha256'] == expected['asset_sha256'] == file_digest(Path(expected['path'])), 'ready asset identity differs')
                log = source / 'logs' / ('glib_' + str(index) + '.log'); contents = log.read_text()
                require('GB_VALIDATE_DONE' in contents and 'Simulation App Shutting Down' in contents, 'native validation/process incomplete')
                lib = S.load_model(job['category'], job['model'], job['scale'], str(source / 'glib'), constraint=job['constraint'], rest_hint_wxyz=job.get('rest_hint_wxyz'))
                require(lib is not None and len(lib.grasps()) > 0, 'no positive native grasps')
                require(Path(lib.path).parent.name == job['id'] and lib.data['dataset'] == job['dataset'], 'native identity differs')
                require(lib.data['constraint'] == job['constraint'] and lib.data.get('legacy_key') == job['legacy_key'], 'native constraint/alias differs')
                S.check_compatible(lib, scale=job['scale'], rest_kind='stable')
                routed = G.v2_entry(job['legacy_key']); require(routed is not None and Path(routed.path) == Path(lib.path), 'native alias resolver differs')
                require(len(G.grasps_from_v2(routed, job['legacy_key'])) > 0, 'native task reader has no positive grasps')
                kept.append({**job, 'rest_kind': 'stable'})
                admitted.append({**base, 'source': str(Path(lib.path).parent), 'library_sha256': file_digest(Path(lib.path)),
                                 'validation_log': str(log), 'validation_log_sha256': file_digest(log), 'positive_grasps': len(lib.grasps()),
                                 'asset_sha256': job['asset_sha256'], 'alias': job['legacy_key'], 'scale': job['scale'], 'constraint': job['constraint'],
                                 'rest_kind': 'stable', 'rest_ids': sorted(lib.rests), 'height_keys': heights, 'arms': ['right', 'left']})
            except (ValueError, OSError, KeyError) as exc:
                skipped.append({**base, 'reason': str(exc)})
        job_path = a.out / (kind + '_admitted_jobs.json'); write_json_atomic(job_path, kept)
        height_path = a.out / (kind + '_heights_by_id.json'); write_json_atomic(height_path, {job['id']: heights for job in kept})
        outputs.append(dict(kind=kind, admitted_manifest=str(job_path), admitted_manifest_sha256=file_digest(job_path),
                            heights_by_id=str(height_path), heights_sha256=file_digest(height_path), source_library_root=str(source / 'glib')))
    report = dict(schema='campaign-held-library-admission-v1', active_policy_sha256=policy_digest(policy), dependency_audit_only=True,
                  reservations=dict(path=str(a.reservations), sha256=file_digest(a.reservations)), asset_hashes=dict(path=str(a.asset_hashes), sha256=file_digest(a.asset_hashes)),
                  tray_floor_proof=dict(path=str(native_trays), sha256=file_digest(native_trays)), admitted=admitted, skipped=skipped, batches=outputs,
                  production_released=False, native_task_geometry_demo_replay_still_required=True)
    write_json_atomic(a.out / 'audit.json', report)
    print(json.dumps(dict(admitted=len(admitted), skipped=skipped, batches=outputs, audit_sha256=file_digest(a.out / 'audit.json'))))


if __name__ == '__main__':
    main()
