#!/usr/bin/env python3
"""Freeze native held-out contact inputs; geometry/rest and contacts gate separately."""
import argparse
import copy
import json
from pathlib import Path

from scripts.campaign_20261110.bind_fresh_candidates import load_contract
from scripts.campaign_20261110.contracts import file_digest, require, write_json_atomic
from scripts.campaign_20261110.group_lane import check_reservations
from scripts.campaign_20261110.reservation_policy import active_policy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'release', 'geometry-root', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    require(not a.out.exists() and a.out.resolve().is_relative_to(a.root.resolve()), 'new private manifest required')
    pointer, policy, hashes = active_policy(a.root)
    from ispatialgym.grasping import schema as S, contact_library as CL
    requests, geometry = [], {}
    selections = [('heldout_small_pairs_v1', {'inter_object_relation_satisfaction-touching_01', 'inter_object_relation_satisfaction-touching_04'}),
                  ('heldout_category_pairs_v1', {'inter_object_relation_satisfaction-touching_small_scale-box_of_crackers-box_of_cookies'}),
                  ('heldout_model_table_tasks_v2', {'alignment-axial_' + str(i).zfill(2) for i in range(1, 6)})]
    for folder, classes in selections:
        path = a.root / folder
        contracts = {row['class_id']: load_contract(row) for row in json.loads((path / 'contract_candidates.json').read_text())['templates']}
        jobs = {row['class_id']: row for row in json.loads((path / 'validation_jobs.json').read_text())['jobs']}
        for cls in sorted(classes):
            template, job = contracts[cls], jobs[cls]
            check_reservations(template, policy, hashes)
            argv = [x.replace('${MATERIALIZED}', str(path)) for x in job['sample_argv']]
            customs = Path(argv[argv.index('--customs') + 1]); entry = json.loads(customs.read_text())[job['alias']]
            axial = template.task == 'alignment-axial'
            for role in ('target',) if axial else ('target', 'partner'):
                obj = next(o for o in template.objects if o.role == role)
                key = obj.category + '/' + obj.model
                require(obj.ready and file_digest(Path(hashes['ready_assets'][key]['path'])) == obj.asset_sha256, 'contact object bytes/readiness differ')
                identity = S.entry_id(obj.category, obj.model, obj.default_scale, 'tabletop', None)
                existing = a.release / 'glib/models' / identity
                target_dir = existing if (existing / 'library.json').is_file() else a.geometry_root / 'glib/models' / identity
                if target_dir != existing:
                    geometry[identity] = dict(id=identity, legacy_key=identity, category=obj.category, model=obj.model,
                        scale=list(obj.default_scale), dataset='behavior-1k-assets', constraint='tabletop', rest_hint_wxyz=None,
                        rest_pose='canonical', kind='tabletop', ready=True, readiness_source=obj.readiness_source,
                        asset_sha256=obj.asset_sha256, campaign_classes=[cls], purpose='native contact geometry and measured stable rest; positive pinch grasp not required for contact-only roles')
                families = ['axial_push', 'axial_brace'] if axial else ['touching_push' if role == 'target' else 'touching_brace']
                params = {}
                if axial:
                    frame = entry['roles']['target']['frame'][key]
                    axis = frame['long'].lstrip('+-'); require(axis in 'xyz' and len(axis) == 1, 'invalid reviewed axial long axis')
                    vector = [0., 0., 0.]; vector['xyz'.index(axis)] = -1. if frame['long'].startswith('-') else 1.
                    params = {family: {'long_axis': vector} for family in families}
                    source = Path(template.scene_setup['reviewed_frame_source'])
                    require(file_digest(source) == template.scene_setup['reviewed_frame_source_sha256'], 'reviewed frame source changed')
                height = job['measured_table_height_key']
                contact_job = dict(id=identity, category=obj.category, model=obj.model, scale=list(obj.default_scale), model_dir=str(target_dir),
                    families=families, subtask=template.task, generation=copy.deepcopy(entry['generation']), params=params, alias=job['alias'] + ' ' + role,
                    validate={'support_z': float(height), 'variant': job['alias']},
                    campaign_provenance=dict(class_id=cls, role=role, template_fingerprint=template.fingerprint,
                        customs=str(customs), customs_sha256=file_digest(customs), asset_sha256=obj.asset_sha256,
                        readiness_source=obj.readiness_source, active_policy_sha256=pointer['policy_sha256']))
                if target_dir == existing:
                    lib, rest, _ = CL.load_object(str(a.release / 'glib'), contact_job)
                    S.check_compatible(lib, scale=obj.default_scale, rest_kind='stable')
                    require(lib.data['category'] == obj.category and lib.data['model'] == obj.model, 'native contact model differs')
                    evidence = dict(library=str(existing / 'library.json'), library_sha256=file_digest(existing / 'library.json'), rest_id=rest)
                else:
                    evidence = dict(status='awaiting fresh native geometry and stable rest')
                requests.append(dict(class_id=cls, role=role, split=template.split, height_key=height, job=contact_job,
                    dependency=evidence, kinds=['push_blade', 'axial_brace'] if axial else ['touch_push' if role == 'target' else 'touch_brace']))
    require(len(requests) == 11, 'expected six touching and five axial contact roles')
    a.out.mkdir(parents=True)
    write_json_atomic(a.out / 'geometry_jobs.json', list(geometry.values()))
    write_json_atomic(a.out / 'contact_profiles.json', dict(active_policy_sha256=pointer['policy_sha256'], requests=requests,
        geometry_manifest_sha256=file_digest(a.out / 'geometry_jobs.json'), native_contact_validation_and_positive_map_required=True, production_released=False))
    print(json.dumps(dict(contact_roles=len(requests), fresh_geometry_jobs=len(geometry),
                         contact_profiles_sha256=file_digest(a.out / 'contact_profiles.json'))))


if __name__ == '__main__':
    main()
