#!/usr/bin/env python3
"""Compile measured, globally reserved small-pair candidates and fresh grasp jobs."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
B1K = Path('/home/minyeongk/Desktop/projects/Behavior-1K-grasp-bands-6a6194600')
sys.path[:0] = [str(ROOT), str(B1K / 'bddl3')]
from contracts import ObjectRef, Template, digest
from readiness_catalog import ReadinessCatalog
from ispatialgym.grasping.manifest import legacy_job
from ispatialgym.grasping.schema import entry_id
from ispatialgym.sampling.bddl_from_iddl import RoleGrounding, render
from ispatialgym.sampling.iddl import parse_file
from ispatialgym.sampling.knowledge import synset_of_category
from ispatialgym.sampling.v2.problem import GroundedProblem, SUBTASK_FAMILY


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_alias(alias):
    for name in ('inside', 'nextto', 'touching'):
        alias = alias.replace(f'satisfaction-{name}-', f'satisfaction-{name}_small_scale-')
    return alias


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('defaults', 'visual-review', 'reservations', 'reserved-asset-hashes', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--measurement-reservations', type=Path)
    p.add_argument('--category-proposal', type=Path)
    p.add_argument('--support-scene-probe', type=Path)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('new immutable output required')
    policy = json.loads(a.reservations.read_text())
    hashes = json.loads(a.reserved_asset_hashes.read_text())
    measured = json.loads(a.defaults.read_text())
    review = json.loads(a.visual_review.read_text())
    measurement_policy = a.measurement_reservations or a.reservations
    if hashes['reservation_sha256'] != sha(a.reservations) or measured['reservation_sha256'] != sha(measurement_policy):
        raise ValueError('policy/measurement mismatch')
    previous = json.loads(measurement_policy.read_text())
    for key in ('model_holdout_models', 'category_holdout_models', 'train_exclude_models'):
        if not set(previous[key]) <= set(policy[key]):
            raise ValueError('measurement policy is not an ancestor')
    for key in ('category_holdout_categories', 'train_exclude_categories', 'category_holdout_rooms'):
        if previous[key] != policy[key]:
            raise ValueError('category/room reservation changed since measurement')
    if measured['status'] != 'native_defaults_and_geometry_recorded' or review['default_geometry_sha256'] != sha(a.defaults):
        raise ValueError('native geometry/review mismatch')
    rows = {r['category'] + '/' + r['model']: r for r in measured['objects']}
    inv = ROOT / 'tmp/campaign_20261110/inventory'
    records = {r['key']: r for r in json.loads((inv / 'cluster_templates_with_scales.json').read_text())['records']}
    source = inv / 'frozen_customs_f24ea424.json'
    originals = json.loads(source.read_text())
    category_mode = a.category_proposal is not None
    if category_mode != (a.support_scene_probe is not None):
        raise ValueError('category mode requires measured private support-scene proof')
    support_proof = json.loads(a.support_scene_probe.read_text()) if category_mode else None
    if category_mode and support_proof['status'] != 'private_support_geometry_and_reload_passed':
        raise ValueError('private support geometry incomplete')
    scene, room, support = ((support_proof['scene'], support_proof['room'], support_proof['object_name']) if category_mode
                            else ('Rs_int', 'living_room_0', 'breakfast_table_skczfi_0'))
    if category_mode and {'scene': scene, 'room': room} not in policy['category_holdout_rooms']:
        raise ValueError('category room is not reserved')
    height_key = support_proof['actual_height_key'] if category_mode else '0.7588'
    scene_path = B1K / 'datasets/2026-challenge-task-instances/scenes' / scene / 'json' / f'{scene}_stable.json'
    init = json.loads(scene_path.read_text())['objects_info']['init_info']
    floor = next(x['args'] for x in init.values() if x['args'].get('category') == 'floors' and room in x['args'].get('in_rooms', []))
    ready = ReadinessCatalog(B1K / 'bddl3/bddl/generated_data')
    scene_roles = {}
    native_support = ({'category': support_proof['model_key'].split('/')[0], 'model': support_proof['model_key'].split('/')[1],
                       'name': support, 'scale': support_proof['default_scale']} if category_mode else init[support]['args'])
    for role, obj in [('floor', floor), ('support', native_support)]:
        status = ready.status(obj['category'], obj['model'])
        usd = B1K / 'datasets/behavior-1k-assets/objects' / obj['category'] / obj['model'] / 'usd' / (obj['model'] + '.encrypted.usd')
        scene_roles[role] = dict(category=obj['category'], model=obj['model'], default_scale=obj['scale'], asset_sha256=sha(usd),
                                 ready=status['ready'], readiness_source=status['readiness_source'], scene_object_name=obj['name'])
    if category_mode and scene_roles['support']['asset_sha256'] != support_proof['asset_sha256']:
        raise ValueError('private support asset proof mismatch')
    support_category = scene_roles['support']['category']
    proposals = policy['model_holdout_templates']
    if category_mode:
        sources = {'inside': 'inter_object_relation_satisfaction-inside_01', 'nextto': 'inter_object_relation_satisfaction-nextto_01',
                   'touching': 'inter_object_relation_satisfaction-touching_01', 'ontop': 'inter_object_relation_satisfaction-ontop_small_scale_01',
                   'under': 'inter_object_relation_satisfaction-under_small_scale_01'}
        proposals = []
        for row in json.loads(a.category_proposal.read_text())['templates']:
            task = row['task']
            if task.startswith('inter_object_relation_satisfaction-') and task.endswith('_small_scale'):
                relation = task.split('-')[1].split('_')[0]
                proposals.append(dict(task=task, class_id=task+'-'+row['target']['category']+'-'+row['partner']['category'],
                    source_class_id=sources[relation], roles=[{**row[role], 'role': role} for role in ('target', 'partner')]))
    definitions, jobs, templates, grasps, blocked = {}, [], [], [], []
    a.out.mkdir(parents=True)
    for proposal in proposals:
        task, cls = proposal['task'], proposal['class_id']
        if not task.startswith('inter_object_relation_satisfaction-') or not task.endswith('_small_scale'):
            continue
        pair = {r['role']: rows.get(r['category'] + '/' + r['model']) for r in proposal['roles'] if r['role'] in {'target', 'partner'}}
        if set(pair) != {'target', 'partner'} or not all(pair.values()):
            continue
        target = pair['target']
        record = records[proposal.get('source_class_id', cls)]
        source_key = record['metadata']['choices']['target']['grasp_key']
        base = legacy_job(source_key, str(ROOT / 'runlogs/grasp_library_20260905'))
        if base is None or base.get('rest_hint_wxyz') is not None or base.get('rest_pose') != 'canonical':
            blocked.append(dict(class_id=cls, reason='new model needs its own native resting pose; source quaternion cannot transfer', source_grasp_key=source_key))
            continue
        role_rows = {**scene_roles, **pair}
        for role, row in role_rows.items():
            key = row['category'] + '/' + row['model']
            if not row['ready']:
                raise ValueError('unready role: ' + key)
            if role != 'floor' and (key not in policy['train_exclude_models'] or hashes['ready_assets'][key]['asset_sha256'] != row['asset_sha256']):
                raise ValueError('role is not globally reserved: ' + key)
            if category_mode and role != 'floor' and row['category'] not in policy['category_holdout_categories']:
                raise ValueError('category candidate role is not globally category-reserved')
            if role in pair and review['objects'][key]['status'] != 'passed':
                raise ValueError('unreviewed pair role: ' + key)
        entry = copy.deepcopy(originals[canonical_alias(record['row']['alias'])])
        iddl = parse_file(str(ROOT / 'ispatialgym/assets/iddl' / entry['iddl']))
        for role, row in pair.items():
            spec = entry['roles'][role]
            spec.update(category=row['category'], representative=row['model'], dataset=row['dataset'], scale={row['category'] + '/' + row['model']: row['default_scale']})
            for key in ('grasp_keys', 'orientation_equivalences', 'rest_pose', 'rest_hint_wxyz'):
                spec.pop(key, None)
        entry['roles']['support']['category'] = support_category
        gkey = entry_id(target['category'], target['model'], target['default_scale'], base['constraint'])
        entry['roles']['target']['grasp_keys'] = {target['category'] + '/' + target['model']: gkey}
        entry['constraints'] = []
        entry['scenes'] = {scene: dict(rooms=[room], bindings={'floor': {'room': room}, 'support': [support],
            'support_surfaces': [dict(name=support, init='ontop', top_z=float(height_key), row='campaign_small_pair_holdout')]},
            whitelist={role: {r['category']: [r['model']]} for role, r in pair.items()}, blacklist={})}
        for key in list(entry):
            if key.startswith('fallback') or key == 'superseded_by':
                entry.pop(key)
        for key in list(entry['generation']):
            if key.startswith('_') or key == 'remove_objects':
                entry['generation'].pop(key)
        # Model-keyed front axes do not transfer to unseen assets. Native geometry
        # derives the selected widest-face relation for each new pair.
        entry['generation'].get('pair_goal_align', {}).pop('front', None)
        entry['generation'].update(templates_max=1, instances_per_template=1)
        entry['roles'] = {r: entry['roles'][r] for r in iddl.roles}
        alias = iddl.variant_template.format(target=pair['target']['category'], partner=pair['partner']['category'], support=support_category)
        if alias in definitions:
            raise ValueError('duplicate native alias')
        did = (28500 if category_mode else 28000) + len(definitions)
        entry['definition_id'] = did
        groundings = {}
        for role in iddl.roles:
            if role == 'floor':
                groundings[role] = RoleGrounding(role, 'scene', same_as='floor')
            else:
                c = role_rows[role]['category']
                groundings[role] = RoleGrounding(role, 'sampleable' if entry['roles'][role].get('sampleable') else 'scene', category=c, synset=synset_of_category(c))
        activity = SUBTASK_FAMILY[task][0]
        rendered = render(iddl, groundings, did, alias, activity)
        bddl = a.out / 'bddl' / activity / f'problem{did}.bddl'
        bddl.parent.mkdir(parents=True, exist_ok=True)
        bddl.write_text(rendered.text)
        entry['campaign_provenance'] = dict(reservation_sha256=sha(a.reservations), defaults_sha256=sha(a.defaults), visual_review_sha256=sha(a.visual_review),
            native_scene_sha256=support_proof['native_scene_sha256'] if category_mode else sha(scene_path), source_customs_sha256=sha(source), source_grasp_metadata_key=source_key, status='candidate_only')
        definitions[alias] = entry
        refs, names = [], {}
        for role, row in role_rows.items():
            instance = rendered.instances[role]
            refs.append(ObjectRef(instance=instance, role=role if role in pair else 'support', category=row['category'], model=row['model'],
                asset_sha256=row['asset_sha256'], default_scale=tuple(row['default_scale']), semantic_kind='floor' if role == 'floor' else 'object', ready=True, readiness_source=row['readiness_source']))
            if row.get('scene_object_name'):
                names[instance] = row['scene_object_name']
        split = 'eval_category' if category_mode else 'eval_model'
        template = Template(template_id=('held_category__' if category_mode else 'held_model__') + cls, task=task, class_id=cls, split=split, scene=scene, room=room, objects=tuple(refs),
            definition_sha256=digest(dict(entry=entry, bddl_sha256=sha(bddl))), scene_setup=dict(task_rooms=[room], inst_to_name=names,
                runtime_bddl_sha256=sha(bddl), customs_alias=alias, reservation_sha256=sha(a.reservations), release_status='candidate_only',
                native_scene_sha256=entry['campaign_provenance']['native_scene_sha256'],
                fresh_metadata_binding_check='required', fresh_demo='pending', fresh_replay='pending', perturbations=False))
        templates.append({**asdict(template), 'fingerprint': template.fingerprint})
        jobs.append(dict(candidate_id=template.template_id, class_id=cls, task=task, alias=alias, scene=scene, room=room, split=split,
            template_fingerprint=template.fingerprint, production_released=False, required_new_grasp_key=gkey, measured_table_height_key=height_key,
            environment={'ISPATIALGYM_STABLE_SCENE_DIR': str(Path(support_proof['native_scene_path']).parent)} if category_mode else {},
            status='requires_positive_fresh_library_and_tabletop_map_then_native_pair_geometry', sample_argv=['python', '-m', 'ispatialgym.sampling.sample_isg',
                '--variant', alias, '--customs', '${MATERIALIZED}/candidate_customs.json', '--bddl-root', '${MATERIALIZED}/bddl', '--scene', scene,
                '--templates', '1', '--instances-per-template', '1', '--max-attempts', '3', '--seed', '20261110', '--no-write-back', '--out', '${CAMPAIGN_OUT}/validation/' + cls]))
        grasps.append(dict(id=gkey, legacy_key=gkey, category=target['category'], model=target['model'], scale=target['default_scale'], dataset=target['dataset'],
            constraint=base['constraint'], rest_hint_wxyz=None, rest_pose='canonical', kind='tabletop', source='campaign_model_holdout_new_model',
            asset_sha256=target['asset_sha256'], ready=True, readiness_source=target['readiness_source'], campaign_classes=[cls], support_height_key=height_key,
            legacy_key_note='New unique routing key only; source model grasps and resting pose are not imported.'))
    custom = a.out / 'candidate_customs.json'
    custom.write_text(json.dumps(definitions, indent=2) + '\n')
    for job in jobs:
        gp = GroundedProblem.load(job['alias'], str(custom), str(a.out / 'bddl'))
        if gp.rooms(scene) != [room]:
            raise ValueError('native room mismatch')
    for filename, value in [('validation_jobs.json', dict(jobs=jobs, blocked=blocked)), ('contract_candidates.json', dict(templates=templates)), ('tabletop_jobs.json', grasps)]:
        (a.out / filename).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(candidates=len(jobs), fresh_libraries=len(grasps), blocked=blocked, manifest_sha256=sha(a.out / 'validation_jobs.json'))))


if __name__ == '__main__':
    main()
