#!/usr/bin/env python3
"""Compile measured hinged-jar category holds on the proved private table."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
B1K = ROOT / 'Behavior-1K'
if not (B1K / 'bddl3').exists():
    B1K = Path('/home/minyeongk/Desktop/projects/Behavior-1K-grasp-bands-6a6194600')
sys.path[:0] = [str(ROOT), str(B1K / 'bddl3')]
from scripts.campaign_20261110.contracts import ObjectRef, Template, digest, require
from scripts.campaign_20261110.group_lane import check_reservations
from scripts.campaign_20261110.readiness_catalog import ReadinessCatalog
from ispatialgym.grasping.artic_motions import motion_kind
from ispatialgym.sampling.bddl_from_iddl import RoleGrounding, render
from ispatialgym.sampling.iddl import parse_file
from ispatialgym.sampling.knowledge import synset_of_category
from ispatialgym.sampling.v2.problem import GroundedProblem


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('inventory', 'defaults', 'visual-review', 'support-scene-probe', 'reservations',
                 'measurement-reservations', 'reserved-asset-hashes', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    require(not a.out.exists(), 'new immutable output required')
    policy, previous, hashes = [json.loads(path.read_text()) for path in
        (a.reservations, a.measurement_reservations, a.reserved_asset_hashes)]
    doc, review, support = [json.loads(path.read_text()) for path in
        (a.defaults, a.visual_review, a.support_scene_probe)]
    require(hashes['reservation_sha256'] == sha(a.reservations), 'reserved hash companion differs')
    require(doc['status'] == 'native_defaults_and_geometry_recorded'
            and doc['reservation_sha256'] == sha(a.measurement_reservations), 'unproved native defaults')
    require(review['default_geometry_sha256'] == sha(a.defaults), 'native visual review differs')
    for field in ('model_holdout_models', 'category_holdout_models', 'train_exclude_models'):
        require(set(previous[field]) <= set(policy[field]), 'measurement reservations were removed')
    for field in ('category_holdout_categories', 'category_holdout_rooms', 'train_exclude_categories'):
        require(previous[field] == policy[field], 'category/room policy changed since measurement')
    require(support['status'] == 'private_support_geometry_and_reload_passed', 'private table proof incomplete')
    scene_path = Path(support['native_scene_path'])
    require(sha(scene_path) == support['native_scene_sha256'], 'private scene bytes changed')
    scene, room, support_name = support['scene'], support['room'], support['object_name']
    require({'scene': scene, 'room': room} in policy['category_holdout_rooms'], 'category room not reserved')
    row = next(r for r in doc['objects'] if (r['category'], r['model']) == ('hinged_jar', 'jnjtrl'))
    key = 'hinged_jar/jnjtrl'; joint = 'j_lid'
    require(row['ready'] and review['objects'][key]['status'] == 'passed', 'hinged jar native review incomplete')
    require(joint in row['joints'] and row['joints'][joint]['upper_limit'] > row['joints'][joint]['lower_limit'], 'native jar hinge missing')
    init = json.loads(scene_path.read_text())['objects_info']['init_info']
    floor = next(r['args'] for r in init.values() if r['args'].get('category') == 'floors' and room in r['args'].get('in_rooms', []))
    native_support = init[support_name]['args']
    require(native_support['scale'] == support['default_scale'], 'private support scale proof differs')
    records = {r['key']: r for r in json.loads((a.inventory / 'cluster_templates_with_scales.json').read_text())['records']}
    source = a.inventory / 'frozen_customs_f24ea424.json'; originals = json.loads(source.read_text())
    ready = ReadinessCatalog(B1K / 'bddl3/bddl/generated_data')
    definitions, templates, jobs, dependencies = {}, [], [], []
    a.out.mkdir(parents=True)
    for direction in ('open', 'close'):
        task = 'configuration_matching-articulation_' + direction + '_small_scale'
        source_class = task + '_01'; cls = task + '-hinged_jar'
        entry = copy.deepcopy(originals[records[source_class]['row']['alias']])
        iddl = parse_file(str(ROOT / 'ispatialgym/assets/iddl' / entry['iddl']))
        entry['roles']['target'].update(category=row['category'], representative=row['model'], dataset=row['dataset'],
            scale={key: row['default_scale']}, joint={key: joint}, handle_key={key: 'hinged_jar__jnjtrl'})
        for name in ('handle_row', 'grasp_keys', 'orientation_equivalences'):
            entry['roles']['target'].pop(name, None)
        entry['roles']['support']['category'] = native_support['category']
        entry['scenes'] = {scene: dict(rooms=[room], bindings={'floor': {'room': room}, 'support': [support_name],
            'support_surfaces': [dict(name=support_name, init='ontop', top_z=float(support['actual_height_key']), row='campaign_category_articulation')]},
            whitelist={'target': {row['category']: [row['model']]}}, blacklist={})}
        entry['constraints'] = []
        for name in list(entry):
            if name.startswith('fallback') or name == 'superseded_by':
                entry.pop(name)
        for name in list(entry['generation']):
            if name.startswith('_') or name == 'remove_objects':
                entry['generation'].pop(name)
        entry['generation'].update(templates_max=1, instances_per_template=1)
        entry['roles'] = {name: entry['roles'][name] for name in iddl.roles}
        alias = iddl.variant_template.format(target=row['category'], support=native_support['category'])
        did = 29600 + len(definitions); entry['definition_id'] = did
        grounds = {}
        for name in iddl.roles:
            if name == 'floor':
                grounds[name] = RoleGrounding(name, 'scene', same_as='floor')
            else:
                category = entry['roles'][name]['category']
                grounds[name] = RoleGrounding(name, 'sampleable' if entry['roles'][name].get('sampleable') else 'scene',
                    category=category, synset=synset_of_category(category))
        rendered = render(iddl, grounds, did, alias, 'configuration_matching')
        bddl = a.out / 'bddl/configuration_matching' / f'problem{did}.bddl'
        bddl.parent.mkdir(parents=True, exist_ok=True); bddl.write_text(rendered.text)
        refs, names = [], {}
        for role, obj in (('floor', floor), ('support', native_support), ('target', row)):
            category, model = obj['category'], obj['model']
            status = ready.status(category, model)
            asset = B1K / 'datasets/behavior-1k-assets/objects' / category / model / 'usd' / (model + '.encrypted.usd')
            asset_sha = sha(asset)
            require(status['ready'], 'category role is not ready')
            if role != 'floor':
                require(category in policy['category_holdout_categories'] and category + '/' + model in policy['category_holdout_models'], 'nonreserved category role')
                require(hashes['ready_assets'][category + '/' + model]['asset_sha256'] == asset_sha, 'reserved role asset changed')
            if role == 'target':
                require(asset_sha == row['asset_sha256'], 'measured jar asset changed')
            refs.append(ObjectRef(instance=rendered.instances[role], role='target' if role == 'target' else 'support', category=category,
                model=model, asset_sha256=asset_sha, default_scale=tuple(obj['default_scale'] if role == 'target' else obj['scale']),
                joint=joint if role == 'target' else None, semantic_kind='floor' if role == 'floor' else 'object', ready=True, readiness_source=status['readiness_source']))
            if role != 'target':
                names[rendered.instances[role]] = obj['name']
        provenance = dict(defaults_sha256=sha(a.defaults), visual_review_sha256=sha(a.visual_review), source_customs_sha256=sha(source),
            support_scene_probe_sha256=sha(a.support_scene_probe), reservation_sha256=sha(a.reservations), native_scene_sha256=sha(scene_path),
            source_class_id=source_class, fresh_geometry='pending', fresh_demo='pending', fresh_replay='pending', production_released=False)
        entry['campaign_provenance'] = provenance; definitions[alias] = entry
        template = Template('held_category__' + cls, task, cls, 'eval_category', scene, room, tuple(refs),
            digest(dict(entry=entry, bddl_sha256=sha(bddl))), {**provenance, 'task_rooms': [room], 'inst_to_name': names,
            'runtime_bddl_sha256': sha(bddl), 'customs_alias': alias, 'native_joint': row['joints'][joint], 'perturbations': False})
        check_reservations(template, policy, hashes)
        templates.append({**asdict(template), 'fingerprint': template.fingerprint})
        environment = {'ISPATIALGYM_STABLE_SCENE_DIR': str(scene_path.parent)}
        jobs.append(dict(candidate_id=template.template_id, class_id=cls, task=task, alias=alias, scene=scene, room=room, split='eval_category',
            template_fingerprint=template.fingerprint, environment=environment, production_released=False, source_manifest=str((a.out / 'validation_jobs.json').resolve()),
            sample_argv=['python', '-m', 'ispatialgym.sampling.sample_isg', '--variant', alias, '--customs', str((a.out / 'candidate_customs.json').resolve()),
            '--bddl-root', str((a.out / 'bddl').resolve()), '--scene', scene, '--templates', '1', '--instances-per-template', '1', '--max-attempts', '3',
            '--seed', '20261110', '--no-write-back', '--out', '${CAMPAIGN_OUT}/validation/' + cls]))
        dependencies.append(dict(class_id=cls, alias=alias, category=row['category'], model=row['model'], dataset=row['dataset'], scale=row['default_scale'],
            asset_sha256=row['asset_sha256'], joint=joint, native_joint=row['joints'][joint], handle_key='hinged_jar__jnjtrl', scene=scene, room=room,
            support_height_key=support['actual_height_key'], operation=motion_kind(entry['generation'], direction), pipeline='small_articulation_motion_prefix',
            direction=direction, generation=entry['generation'], environment=environment, ready=True, readiness_source=row['readiness_source'],
            status='dependency_request_not_validated'))
    custom = a.out / 'candidate_customs.json'; custom.write_text(json.dumps(definitions, indent=2) + '\n')
    for job in jobs:
        problem = GroundedProblem.load(job['alias'], str(custom), str(a.out / 'bddl'))
        require(problem.rooms(scene) == [room], 'native category room mismatch')
    for filename, value in [('contract_candidates.json', {'templates': templates}), ('validation_jobs.json', {'jobs': jobs}),
                            ('articulation_dependency_requests.json', dependencies)]:
        (a.out / filename).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(candidates=len(jobs), manifest_sha256=sha(a.out / 'validation_jobs.json'))))


if __name__ == '__main__':
    main()
