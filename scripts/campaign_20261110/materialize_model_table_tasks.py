#!/usr/bin/env python3
"""Freeze ready model-held axial and attachment candidates on a measured table."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys

from scripts.campaign_20261110.contracts import ObjectRef, Template, digest, file_digest, require, write_json_atomic
from scripts.campaign_20261110.group_lane import check_reservations
from scripts.campaign_20261110.readiness_catalog import ReadinessCatalog


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'snapshot', 'inventory', 'defaults', 'visual-review', 'table-proof', 'reservations', 'reserved-asset-hashes', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    require(a.out.resolve().is_relative_to(a.root.resolve()) and not a.out.exists(), 'new owned output required')
    policy, hashes, defaults, review, table_proof = [json.loads(path.read_text()) for path in
        (a.reservations, a.reserved_asset_hashes, a.defaults, a.visual_review, a.table_proof)]
    require(hashes['reservation_sha256'] == file_digest(a.reservations) == defaults['reservation_sha256'], 'policy/native identity differs')
    require(defaults['status'] == 'native_defaults_and_geometry_recorded' and review['default_geometry_sha256'] == file_digest(a.defaults), 'native geometry/review differs')
    require(table_proof['status'] == 'native_tray_floor_measurement_passed', 'native table measurement incomplete')
    b1k = a.snapshot / 'Behavior-1K'
    scene, room, support_name = table_proof['scene'], table_proof['room'], table_proof['support_name']
    scene_path = b1k / 'datasets/2026-challenge-task-instances/scenes' / scene / 'json' / (scene + '_stable.json')
    require(file_digest(scene_path) == table_proof['native_scene_sha256'], 'measured table scene changed')
    init = json.loads(scene_path.read_text())['objects_info']['init_info']
    support = init[support_name]['args']
    require(support['scale'] == table_proof['support_scale'], 'native table scale differs')
    floor = next(row['args'] for row in init.values() if row['args'].get('category') == 'floors' and room in row['args'].get('in_rooms', []))
    catalog = ReadinessCatalog(b1k / 'bddl3/bddl/generated_data')
    scene_roles = {}
    for role, obj in (('support', support), ('floor', floor)):
        status = catalog.status(obj['category'], obj['model']); require(status['ready'], 'unready scene role')
        asset = b1k / 'datasets/behavior-1k-assets/objects' / obj['category'] / obj['model'] / 'usd' / (obj['model'] + '.encrypted.usd')
        scene_roles[role] = dict(category=obj['category'], model=obj['model'], default_scale=obj['scale'], asset_sha256=file_digest(asset),
                                ready=True, readiness_source=status['readiness_source'], scene_object_name=obj['name'])
    measured = {row['category'] + '/' + row['model']: row for row in defaults['objects']}
    source = a.inventory / 'frozen_customs_f24ea424.json'
    originals = json.loads(source.read_text())
    records = {row['key']: row for row in json.loads((a.inventory / 'cluster_templates_with_scales.json').read_text())['records']}
    from ispatialgym.grasping.schema import entry_id
    from ispatialgym.sampling.bddl_from_iddl import RoleGrounding, render
    from ispatialgym.sampling.iddl import parse_file
    from ispatialgym.sampling.knowledge import synset_of_category
    from ispatialgym.sampling.v2.problem import GroundedProblem, iter_template_choices
    import numpy as np
    proposals = [row for row in policy['model_holdout_templates'] if row['task'] in ('alignment-axial', 'configuration_matching-attachment')]
    require(len(proposals) == 9, 'expected five axial and four attachment classes')
    a.out.mkdir(parents=True)
    templates, jobs, grasps, dependencies = [], [], [], []
    for index, proposal in enumerate(proposals):
        task, cls = proposal['task'], proposal['class_id']
        roles = {**scene_roles, **{row['role']: measured[row['category'] + '/' + row['model']] for row in proposal['roles'] if row['role'] in ('target', 'partner')}}
        for role, row in roles.items():
            key = row['category'] + '/' + row['model']; require(row['ready'], 'unready model role')
            if role != 'floor':
                require(key in policy['model_holdout_models'] and row['asset_sha256'] == hashes['ready_assets'][key]['asset_sha256'], 'model role reservation differs')
            if role in ('target', 'partner'):
                require(review['objects'][key]['status'] == 'passed' and file_digest(Path(row['asset_path'])) == row['asset_sha256'], 'unreviewed or changed native role')
        target = roles['target']; record = records[cls]; entry = copy.deepcopy(originals[record['row']['alias']])
        iddl = parse_file(str(a.snapshot / 'ispatialgym/assets/iddl' / entry['iddl']))
        for role in ('target', 'partner') if 'partner' in roles else ('target',):
            row = roles[role]; key = row['category'] + '/' + row['model']
            entry['roles'][role].update(category=row['category'], representative=row['model'], dataset=row['dataset'], scale={key: row['default_scale']})
            for name in ('grasp_keys', 'orientation_equivalences', 'rest_pose', 'rest_hint_wxyz'):
                entry['roles'][role].pop(name, None)
        entry['roles']['support']['category'] = support['category']
        gkey = entry_id(target['category'], target['model'], target['default_scale'], 'attachment_free') if 'partner' in roles else None
        if gkey:
            entry['roles']['target']['grasp_keys'] = {target['category'] + '/' + target['model']: gkey}
        entry['scenes'] = {scene: dict(rooms=[room], bindings={'floor': {'room': room}, 'support': [support_name],
            'support_surfaces': [dict(name=support_name, init='ontop', top_z=float(table_proof['tabletop_height_key']), row='campaign_model_table_tasks')]},
            whitelist={role: {row['category']: [row['model']]} for role, row in roles.items() if role in ('target', 'partner')}, blacklist={})}
        entry['constraints'] = []
        for name in list(entry):
            if name.startswith('fallback') or name == 'superseded_by': entry.pop(name)
        for name in list(entry['generation']):
            if name.startswith('_') or name == 'remove_objects': entry['generation'].pop(name)
        entry['generation'].update(templates_max=1, instances_per_template=1)
        entry['roles'] = {name: entry['roles'][name] for name in iddl.roles}
        alias = iddl.variant_template.format(target=target['category'], partner=roles.get('partner', {}).get('category', ''), support=support['category'])
        did = 30800 + index; entry['definition_id'] = did
        groundings = {role: RoleGrounding(role, 'scene', same_as='floor') if role == 'floor' else
                      RoleGrounding(role, 'sampleable' if entry['roles'][role].get('sampleable') else 'scene',
                                    category=roles[role]['category'], synset=synset_of_category(roles[role]['category'])) for role in iddl.roles}
        rendered = render(iddl, groundings, did, alias, task.split('-')[0])
        directory = a.out / cls; directory.mkdir()
        bddl = directory / 'bddl' / task.split('-')[0] / ('problem' + str(did) + '.bddl')
        bddl.parent.mkdir(parents=True); bddl.write_text(rendered.text)
        refs, names = [], {}
        for role, row in roles.items():
            instance = rendered.instances[role]
            refs.append(ObjectRef(instance=instance, role=role if role in ('target', 'partner') else 'support', category=row['category'], model=row['model'],
                asset_sha256=row['asset_sha256'], default_scale=tuple(row['default_scale']), semantic_kind='floor' if role == 'floor' else 'object',
                ready=True, readiness_source=row['readiness_source']))
            if row.get('scene_object_name'): names[instance] = row['scene_object_name']
        proof = dict(defaults_sha256=file_digest(a.defaults), visual_review_sha256=file_digest(a.visual_review), table_proof_sha256=file_digest(a.table_proof),
                     native_scene_sha256=file_digest(scene_path), source_customs_sha256=file_digest(source), source_class_id=cls,
                     source_support_adaptation='Existing globally reserved functional table with native measured height and exact scene default scale',
                     reservation_sha256=file_digest(a.reservations), production_released=False)
        entry['campaign_provenance'] = proof
        customs = directory / 'candidate_customs.json'; write_json_atomic(customs, {alias: entry})
        template = Template('held_model__' + cls, task, cls, 'eval_model', scene, room, tuple(refs), digest(dict(entry=entry, bddl_sha256=file_digest(bddl))),
                            {**proof, 'task_rooms': [room], 'inst_to_name': names, 'runtime_bddl_sha256': file_digest(bddl), 'customs_alias': alias,
                             'fresh_geometry': 'pending', 'fresh_demo': 'pending', 'fresh_replay': 'pending', 'perturbations': False})
        check_reservations(template, policy, hashes)
        gp = GroundedProblem.load(alias, str(customs), str(directory / 'bddl'))
        choices = list(iter_template_choices(gp, scene, np.random.default_rng(20261110), max_templates=1))
        require(len(choices) == 1, 'native model choice unavailable for ' + cls)
        for role in ('target', 'partner') if 'partner' in roles else ('target',):
            require(choices[0].models[role].model == roles[role]['model'], 'native role choice differs')
        templates.append({**asdict(template), 'fingerprint': template.fingerprint})
        jobs.append(dict(candidate_id=template.template_id, class_id=cls, task=task, alias=alias, scene=scene, room=room, split='eval_model',
                         template_fingerprint=template.fingerprint, required_new_grasp_key=gkey, measured_table_height_key=table_proof['tabletop_height_key'],
                         source_manifest=str((a.out / 'validation_jobs.json').resolve()), production_released=False,
                         sample_argv=[sys.executable, '-m', 'ispatialgym.sampling.sample_isg', '--variant', alias, '--customs', str(customs),
                                      '--bddl-root', str(directory / 'bddl'), '--scene', scene, '--templates', '1', '--instances-per-template', '1',
                                      '--max-attempts', '3', '--seed', '20261110', '--no-write-back', '--out', '${CAMPAIGN_OUT}/validation/' + cls]))
        request = dict(class_id=cls, target=target['category'] + '/' + target['model'], scale=target['default_scale'], asset_sha256=target['asset_sha256'],
                       height_key=table_proof['tabletop_height_key'], ready=True, readiness_source=target['readiness_source'], status='native_dependency_pending')
        if gkey:
            metadata = {}
            for role in ('target', 'partner'):
                row = roles[role]; path = b1k / 'datasets/behavior-1k-assets/objects' / row['category'] / row['model'] / 'misc/metadata.json'
                links = json.loads(path.read_text()).get('meta_links', {})
                metadata[role] = dict(path=str(path), sha256=file_digest(path), attachment_links={key: value['attachment'] for key, value in links.items() if value.get('attachment')})
            require(all(row['attachment_links'] for row in metadata.values()), 'native attachment links absent')
            request.update(pipeline='attachment', grasp_key=gkey, partner=roles['partner']['category'] + '/' + roles['partner']['model'],
                           partner_scale=roles['partner']['default_scale'], native_metalink_evidence=metadata)
            grasps.append(dict(id=gkey, legacy_key=gkey, category=target['category'], model=target['model'], scale=target['default_scale'], dataset=target['dataset'],
                               constraint='attachment_free', rest_pose='canonical', rest_hint_wxyz=None, kind='tabletop', asset_sha256=target['asset_sha256'],
                               ready=True, readiness_source=target['readiness_source'], campaign_classes=[cls], source='campaign_model_attachment'))
        else:
            request.update(pipeline='contact', kinds=[dict(kind='push_blade', arm='right'), dict(kind='axial_brace', arm='left')],
                           native_rest='Measure new model; do not transfer original model contacts or resting quaternion')
        dependencies.append(request)
    for filename, value in [('contract_candidates.json', dict(templates=templates)), ('validation_jobs.json', dict(jobs=jobs)),
                            ('tabletop_jobs.json', grasps), ('native_dependency_requests.json', dependencies)]:
        write_json_atomic(a.out / filename, value)
    print(json.dumps(dict(candidates=len(jobs), manifest_sha256=file_digest(a.out / 'validation_jobs.json'))))


if __name__ == '__main__':
    main()
