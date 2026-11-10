#!/usr/bin/env python3
"""Compile clock-on-table and armchair-next-to-table category geometry gates."""
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
from ispatialgym.grasping.schema import entry_id
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
    p.add_argument('--relations', nargs='+', choices=['ontop', 'nextto', 'under'], default=['ontop', 'nextto'])
    a = p.parse_args()
    require(not a.out.exists(), 'new immutable output required')
    policy, previous, hashes = [json.loads(path.read_text()) for path in (a.reservations, a.measurement_reservations, a.reserved_asset_hashes)]
    doc, review, support = [json.loads(path.read_text()) for path in (a.defaults, a.visual_review, a.support_scene_probe)]
    require(hashes['reservation_sha256'] == sha(a.reservations), 'reservation hash companion changed')
    require(doc['status'] == 'native_defaults_and_geometry_recorded' and doc['reservation_sha256'] == sha(a.measurement_reservations), 'native default proof differs')
    require(review['default_geometry_sha256'] == sha(a.defaults), 'visual proof differs')
    for key in ('model_holdout_models', 'category_holdout_models', 'train_exclude_models'):
        require(set(previous[key]) <= set(policy[key]), 'nonmonotonic measurement policy')
    for key in ('category_holdout_categories', 'category_holdout_rooms', 'train_exclude_categories'):
        require(previous[key] == policy[key], 'category or room policy changed')
    require(support['status'] == 'private_support_geometry_and_reload_passed', 'private table proof incomplete')
    scene_path = Path(support['native_scene_path']); require(sha(scene_path) == support['native_scene_sha256'], 'private scene changed')
    scene, room, name = support['scene'], support['room'], support['object_name']
    init = json.loads(scene_path.read_text())['objects_info']['init_info']; partner = init[name]['args']
    require(partner['scale'] == support['default_scale'], 'table scale proof differs')
    floor = next(x['args'] for x in init.values() if x['args'].get('category') == 'floors' and room in x['args'].get('in_rooms', []))
    rows = {x['category'] + '/' + x['model']: x for x in doc['objects']}
    source = a.inventory / 'frozen_customs_f24ea424.json'; originals = json.loads(source.read_text())
    records = {x['key']: x for x in json.loads((a.inventory / 'cluster_templates_with_scales.json').read_text())['records']}
    ready = ReadinessCatalog(B1K / 'bddl3/bddl/generated_data')
    definitions, templates, jobs, grasp_jobs, dependencies = {}, [], [], [], []
    a.out.mkdir(parents=True)
    require(len(a.relations) == len(set(a.relations)), 'duplicate requested relation')
    for relation in a.relations:
        model_key = {'ontop': 'alarm_clock/vqwovi', 'nextto': 'armchair/pmrcex', 'under': 'shoe_box/pitwey'}[relation]
        target = rows[model_key]; require(target['ready'] and review['objects'][model_key]['status'] == 'passed', 'target visual/default evidence incomplete')
        task = 'inter_object_relation_satisfaction-' + relation + '_large_scale'; source_class = task + ('_03' if relation == 'under' else '_01')
        cls = task + '-' + target['category'] + '-' + partner['category']
        entry = copy.deepcopy(originals[records[source_class]['row']['alias']]); iddl = parse_file(str(ROOT / 'ispatialgym/assets/iddl' / entry['iddl']))
        entry['roles']['target'].update(category=target['category'], representative=target['model'], dataset=target['dataset'], scale={model_key: target['default_scale']})
        for key in ('orientation_equivalences', 'grasp_keys', 'rest_pose', 'rest_hint_wxyz'):
            entry['roles']['target'].pop(key, None)
        entry['roles']['partner']['category'] = partner['category']
        gkey = entry_id(target['category'], target['model'], target['default_scale'], 'tabletop') if relation in ('ontop', 'under') else None
        if gkey:
            entry['roles']['target']['grasp_keys'] = {model_key: gkey}
        entry['scenes'] = {scene: dict(rooms=[room], bindings={'floor': {'room': room}, 'support': {'room': room}, 'partner': [name]},
            whitelist={'target': {target['category']: [target['model']]}}, blacklist={})}
        entry['constraints'] = []
        for key in list(entry):
            if key.startswith('fallback') or key == 'superseded_by':
                entry.pop(key)
        for key in list(entry['generation']):
            if key.startswith('_') or key == 'remove_objects':
                entry['generation'].pop(key)
        entry['generation'].update(templates_max=1, instances_per_template=1)
        entry['roles'] = {key: entry['roles'][key] for key in iddl.roles}
        alias = iddl.variant_template.format(target=target['category'], partner=partner['category'], support='floor')
        did = 29700 + len(definitions); entry['definition_id'] = did
        grounds = {key: RoleGrounding(key, 'sampleable' if key == 'target' else 'scene', category=entry['roles'][key]['category'],
            synset=synset_of_category(entry['roles'][key]['category'])) for key in iddl.roles if key != 'floor'}
        grounds['floor'] = RoleGrounding('floor', 'scene', same_as='floor')
        rendered = render(iddl, grounds, did, alias, 'inter_object_relation_satisfaction')
        bddl = a.out / 'bddl/inter_object_relation_satisfaction' / f'problem{did}.bddl'; bddl.parent.mkdir(parents=True, exist_ok=True); bddl.write_text(rendered.text)
        refs, names = [], {}
        for role, obj in [('target', target), ('partner', partner), ('support', floor), ('floor', floor)]:
            category, model = obj['category'], obj['model']; status = ready.status(category, model)
            asset = B1K / 'datasets/behavior-1k-assets/objects' / category / model / 'usd' / (model + '.encrypted.usd'); asset_sha = sha(asset)
            require(status['ready'], 'unready category role')
            if category != 'floors':
                require(category in policy['category_holdout_categories'] and category + '/' + model in policy['category_holdout_models'], 'unreserved category role')
                require(asset_sha == hashes['ready_assets'][category + '/' + model]['asset_sha256'], 'category role USD changed')
            if role == 'target':
                require(asset_sha == target['asset_sha256'], 'measured target USD changed')
            refs.append(ObjectRef(instance=rendered.instances[role], role=role if role in ('target', 'partner') else 'support', category=category, model=model,
                asset_sha256=asset_sha, default_scale=tuple(obj['default_scale'] if role == 'target' else obj['scale']), semantic_kind='floor' if category == 'floors' else 'object',
                ready=True, readiness_source=status['readiness_source']))
            if role != 'target':
                names[rendered.instances[role]] = obj['name']
        proof = dict(defaults_sha256=sha(a.defaults), visual_review_sha256=sha(a.visual_review), source_customs_sha256=sha(source), source_class_id=source_class,
            support_scene_probe_sha256=sha(a.support_scene_probe), reservation_sha256=sha(a.reservations), native_scene_sha256=sha(scene_path), production_released=False)
        entry['campaign_provenance'] = proof; definitions[alias] = entry
        template = Template('held_category__' + cls, task, cls, 'eval_category', scene, room, tuple(refs), digest(dict(entry=entry, bddl_sha256=sha(bddl))),
            {**proof, 'task_rooms': [room], 'inst_to_name': names, 'runtime_bddl_sha256': sha(bddl), 'customs_alias': alias,
             'fresh_geometry': 'pending', 'fresh_demo': 'pending', 'fresh_replay': 'pending', 'perturbations': False})
        check_reservations(template, policy, hashes); templates.append({**asdict(template), 'fingerprint': template.fingerprint})
        environment = {'ISPATIALGYM_STABLE_SCENE_DIR': str(scene_path.parent)}
        jobs.append(dict(candidate_id=template.template_id, class_id=cls, task=task, alias=alias, scene=scene, room=room, split='eval_category',
            template_fingerprint=template.fingerprint, environment=environment, required_new_grasp_key=gkey, production_released=False,
            source_manifest=str((a.out / 'validation_jobs.json').resolve()), status='requires_fresh_grasps_and_floor_and_table_maps' if gkey else 'requires_native_floor_push_geometry',
            sample_argv=['python', '-m', 'ispatialgym.sampling.sample_isg', '--variant', alias, '--customs', str((a.out / 'candidate_customs.json').resolve()), '--bddl-root',
            str((a.out / 'bddl').resolve()), '--scene', scene, '--templates', '1', '--instances-per-template', '1', '--max-attempts', '3', '--seed', '20261110',
            '--no-write-back', '--out', '${CAMPAIGN_OUT}/validation/' + cls]))
        if gkey:
            grasp_jobs.append(dict(id=gkey, legacy_key=gkey, category=target['category'], model=target['model'], scale=target['default_scale'], dataset=target['dataset'],
                constraint='tabletop', rest_pose='canonical', rest_hint_wxyz=None, kind='tabletop', asset_sha256=target['asset_sha256'], ready=True,
                readiness_source=target['readiness_source'], campaign_classes=[cls], source='campaign_category_floor_pair'))
            dependencies.append(dict(class_id=cls, grasp_key=gkey, tabletop_height_key=support['actual_height_key'] if relation == 'ontop' else None, floor_height_key='0.0000', arms=['right', 'left'],
                native_rest_geometry='pending', map_validation='pending'))
    custom = a.out / 'candidate_customs.json'; custom.write_text(json.dumps(definitions, indent=2) + '\n')
    for job in jobs:
        problem = GroundedProblem.load(job['alias'], str(custom), str(a.out / 'bddl'))
        require(problem.rooms(scene) == [room], 'native category room mismatch')
    for filename, value in [('contract_candidates.json', {'templates': templates}), ('validation_jobs.json', {'jobs': jobs}),
                            ('tabletop_jobs.json', grasp_jobs), ('map_dependency_requests.json', dependencies)]:
        (a.out / filename).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(candidates=len(jobs), manifest_sha256=sha(a.out / 'validation_jobs.json'))))


if __name__ == '__main__':
    main()
