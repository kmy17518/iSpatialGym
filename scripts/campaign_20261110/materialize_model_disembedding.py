#!/usr/bin/env python3
"""Freeze nine exact held-model disembedding variants and bounded full pools."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path

from scripts.campaign_20261110.binding_pools import BindingPool, PoolCandidate, bind_variant, write_pool
from scripts.campaign_20261110.contracts import ObjectRef, Template, digest, file_digest, require, write_json_atomic
from scripts.campaign_20261110.group_lane import check_reservations
from scripts.campaign_20261110.readiness_catalog import ReadinessCatalog


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'snapshot', 'inventory', 'defaults', 'visual-review', 'tray-floor-proof', 'reservations', 'reserved-asset-hashes', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--support-scene-proof', type=Path)
    p.add_argument('--measurement-reservations', type=Path)
    a = p.parse_args()
    require(not a.out.exists() and a.out.resolve().is_relative_to(a.root.resolve()), 'new owned output required')
    policy, hashes, defaults, review, trays = [json.loads(path.read_text()) for path in
        (a.reservations, a.reserved_asset_hashes, a.defaults, a.visual_review, a.tray_floor_proof)]
    require(hashes['reservation_sha256'] == file_digest(a.reservations), 'reservation asset identity differs')
    measured_policy = a.measurement_reservations or a.reservations
    require(defaults['reservation_sha256'] == file_digest(measured_policy), 'native measurement policy differs')
    ancestor = json.loads(measured_policy.read_text())
    for field in ('model_holdout_models', 'category_holdout_models', 'train_exclude_models'):
        require(set(ancestor[field]) <= set(policy[field]), 'native measurement policy is not an ancestor')
    for field in ('category_holdout_categories', 'train_exclude_categories'):
        require(set(ancestor[field]) <= set(policy[field]), 'category reservation removed since native measurement')
    require({(row['scene'], row['room']) for row in ancestor['category_holdout_rooms']} <=
            {(row['scene'], row['room']) for row in policy['category_holdout_rooms']},
            'room reservation removed since native measurement')
    require(defaults['status'] == 'native_defaults_and_geometry_recorded' and review['default_geometry_sha256'] == file_digest(a.defaults), 'native review identity differs')
    require(trays['status'] == 'native_tray_floor_measurement_passed', 'tray floor proof incomplete')
    b1k = a.snapshot / 'Behavior-1K'; scene, room = trays['scene'], trays['room']
    scene_path = b1k / 'datasets/2026-challenge-task-instances/scenes' / scene / 'json' / (scene + '_stable.json')
    scene_environment = {}
    if a.support_scene_proof:
        support_proof = json.loads(a.support_scene_proof.read_text())
        require(a.support_scene_proof.resolve().is_relative_to(a.root.resolve()), 'support proof outside campaign')
        require(support_proof['status'] == 'private_support_geometry_and_reload_passed'
                and support_proof['split'] == 'eval_model' and support_proof['original_object_values_unchanged'],
                'private support scene lacks native geometry/reload proof')
        require((support_proof['scene'], support_proof['room'], support_proof['object_name']) ==
                (scene, room, trays['support_name']), 'support/tray context differs')
        require(file_digest(scene_path) == support_proof['source_scene_sha256'], 'original support context changed')
        scene_path = Path(support_proof['native_scene_path']).resolve()
        require(scene_path.is_relative_to(a.root.resolve())
                and file_digest(scene_path) == support_proof['native_scene_sha256'], 'private support scene changed')
        scene_environment = {'ISPATIALGYM_STABLE_SCENE_DIR': str(scene_path.parent)}
    require(file_digest(scene_path) == trays['native_scene_sha256'], 'native support scene changed')
    init = json.loads(scene_path.read_text())['objects_info']['init_info']; support = init[trays['support_name']]['args']
    require(support['scale'] == trays['support_scale'], 'measured support scale differs')
    floor = next(row['args'] for row in init.values() if row['args'].get('category') == 'floors' and room in row['args'].get('in_rooms', []))
    measured = {row['category'] + '/' + row['model']: row for row in defaults['objects']}
    for model in ('gsxbym', 'mhhoga'):
        native = trays['objects'][model]; row = measured['tray/' + model]
        require(native['asset_sha256'] == row['asset_sha256'] and native['default_scale'] == row['default_scale'], 'native tray-floor identity differs')
        require(native['map_height_key'] == f"{native['map_height_above_floor']:.4f}", 'native tray height key differs')
    catalog = ReadinessCatalog(b1k / 'bddl3/bddl/generated_data'); scene_roles = {}
    for role, obj in (('support', support), ('floor', floor)):
        status = catalog.status(obj['category'], obj['model']); require(status['ready'], 'unready scene role')
        asset = b1k / 'datasets/behavior-1k-assets/objects' / obj['category'] / obj['model'] / 'usd' / (obj['model'] + '.encrypted.usd')
        scene_roles[role] = dict(category=obj['category'], model=obj['model'], default_scale=obj['scale'], asset_sha256=file_digest(asset),
                                ready=True, readiness_source=status['readiness_source'], scene_object_name=obj['name'])
    source = a.inventory / 'frozen_customs_f24ea424.json'; originals = json.loads(source.read_text())
    records = {row['key']: row for row in json.loads((a.inventory / 'cluster_templates_with_scales.json').read_text())['records']}
    from ispatialgym.grasping.schema import entry_id
    from ispatialgym.sampling.bddl_from_iddl import RoleGrounding, render
    from ispatialgym.sampling.iddl import parse_file
    from ispatialgym.sampling.knowledge import synset_of_category
    from ispatialgym.sampling.v2.problem import GroundedProblem, iter_template_choices
    import numpy as np
    proposals = [row for row in policy['model_holdout_templates'] if row['task'] == 'disembedding' and row['class_id'] != 'disembedding_01']
    require(len(proposals) == 9, 'expected nine variable-count held classes; LEF remains fixed')
    a.out.mkdir(parents=True); templates, jobs, grasp_jobs, requests = [], [], [], []
    for index, proposal in enumerate(proposals):
        cls = proposal['class_id']; actors = proposal['roles']
        target_key = next(row['category'] + '/' + row['model'] for row in actors if row['role'] == 'target')
        # The category list is a pool. The native task excludes target-identical
        # distractors; choose one copy of each other admitted model initially.
        distractor_keys = sorted({row['category'] + '/' + row['model'] for row in actors if row['role'] == 'distractors'} - {target_key})
        require(distractor_keys, 'no distinct ready distractor model remains')
        role_rows = {**scene_roles, 'target': measured[target_key], 'source': measured['tray/gsxbym'], 'destination': measured['tray/mhhoga']}
        distractors = [measured[key] for key in distractor_keys]
        for role, row in list(role_rows.items()) + [('distractors', row) for row in distractors]:
            key = row['category'] + '/' + row['model']; require(row['ready'], 'unready held role')
            if role != 'floor':
                require(key in policy['model_holdout_models'] and row['asset_sha256'] == hashes['ready_assets'][key]['asset_sha256'], 'held model/hash differs')
            if role not in ('floor', 'support'):
                require(review['objects'][key]['status'] == 'passed' and file_digest(Path(row['asset_path'])) == row['asset_sha256'], 'native role review or USD differs')
        entry = copy.deepcopy(originals[records[cls]['row']['alias']]); source_definition = digest(entry)
        require(not entry['generation'].get('fixed_layout'), 'fixed puzzle cannot use variable counts')
        iddl = parse_file(str(a.snapshot / 'ispatialgym/assets/iddl' / entry['iddl']))
        for role in ('target', 'source', 'destination'):
            row = role_rows[role]; key = row['category'] + '/' + row['model']
            entry['roles'][role].update(category=row['category'], representative=row['model'], dataset=row['dataset'], scale={key: row['default_scale']})
            for name in ('grasp_keys', 'orientation_equivalences', 'rest_pose', 'rest_hint_wxyz'): entry['roles'][role].pop(name, None)
        target = role_rows['target']; gkey = entry_id(target['category'], target['model'], target['default_scale'], 'tabletop')
        entry['roles']['target']['grasp_keys'] = {target_key: gkey}
        entry['roles']['support']['category'] = support['category']
        entry['roles']['distractors'].update(category=[row['category'] for row in distractors], scale={key: measured[key]['default_scale'] for key in distractor_keys})
        for name in ('representative', 'grasp_keys', 'orientation_equivalences', 'rest_pose', 'rest_hint_wxyz'): entry['roles']['distractors'].pop(name, None)
        members = {row['category']: 1 for row in distractors}; require(len(members) == len(distractors), 'initial choice needs one model per category')
        entry['generation']['groups']['distractors'] = dict(size=len(distractors), draw='same_model', members=members)
        budget = float(entry['generation'].get('tray_budget', .85))
        entry['generation']['distractor_budget'] = dict(budget_fraction=budget, n_distractors=len(distractors), source='Initial exact variant; coverage requires native profile')
        for name in list(entry['generation']):
            if name.startswith('_') or name in ('remove_objects', 'fixed_layout'): entry['generation'].pop(name)
        for name in list(entry):
            if name.startswith('fallback') or name == 'superseded_by': entry.pop(name)
        entry['generation'].update(templates_max=1, instances_per_template=1); entry['constraints'] = []
        whitelist = {role: {role_rows[role]['category']: [role_rows[role]['model']]} for role in ('target', 'source', 'destination')}
        whitelist['distractors'] = {row['category']: [row['model']] for row in distractors}
        entry['scenes'] = {scene: dict(rooms=[room], bindings={'floor': {'room': room}, 'support': [support['name']],
            'support_surfaces': [dict(name=support['name'], init='ontop', top_z=float(trays['tabletop_height_key']), row='campaign_model_disembedding')]},
            whitelist=whitelist, blacklist={})}
        entry['roles'] = {name: entry['roles'][name] for name in iddl.roles}
        alias = iddl.variant_template.format(target=target['category'], distractors='_'.join(row['category'] + '_1' for row in distractors), support=support['category'])
        did = 30900 + index; entry['definition_id'] = did; grounds = {}
        for role in iddl.roles:
            if role == 'floor': grounds[role] = RoleGrounding(role, 'scene', same_as='floor')
            elif role == 'distractors': grounds[role] = RoleGrounding(role, 'group', members=[(row['category'], synset_of_category(row['category'])) for row in distractors])
            else:
                category = role_rows[role]['category']; grounds[role] = RoleGrounding(role, 'sampleable' if entry['roles'][role].get('sampleable') else 'scene', category=category, synset=synset_of_category(category))
        rendered = render(iddl, grounds, did, alias, 'disembedding'); directory = a.out / cls; directory.mkdir()
        bddl = directory / 'bddl/disembedding' / ('problem' + str(did) + '.bddl'); bddl.parent.mkdir(parents=True); bddl.write_text(rendered.text)
        refs, names = [], {}
        pairs = [(role, row, rendered.instances[role]) for role, row in role_rows.items()]
        pairs += [('distractors', row, instance) for row, instance in zip(distractors, rendered.instances['distractors.members'])]
        for role, row, instance in pairs:
            refs.append(ObjectRef(instance=instance, role='target' if role == 'target' else 'distractor' if role == 'distractors' else 'support',
                category=row['category'], model=row['model'], asset_sha256=row['asset_sha256'], default_scale=tuple(row['default_scale']),
                semantic_kind='floor' if role == 'floor' else 'tray' if role in ('source', 'destination') else 'object', ready=True, readiness_source=row['readiness_source']))
            if row.get('scene_object_name'): names[instance] = row['scene_object_name']
        choices = tuple(PoolCandidate('model_' + digest(dict(role=obj.role, category=obj.category, model=obj.model, sha=obj.asset_sha256, scale=obj.default_scale))[:20], obj,
                                      1 if obj.role == 'target' else 48) for obj in refs if obj.role != 'support')
        pool = BindingPool('model_disembedding_pool__' + cls + '_v1', cls, 'eval_model', scene, room, tuple(obj for obj in refs if obj.role == 'support'), choices,
                           {'target': (1, 1), 'distractor': (1, 48)}, source_definition, task_rooms=(room,), provenance=dict(defaults_sha256=file_digest(a.defaults),
                           visual_review_sha256=file_digest(a.visual_review), tray_floor_probe_sha256=file_digest(a.tray_floor_proof),
                           capacity_note='Finite maximum48; no native packing or bucket-coverage claim', target_identical_distractors_excluded=target_key))
        check_reservations(pool.audit_template(), policy, hashes); pool_path = write_pool(directory / 'binding_pool.json', pool)
        proof = dict(defaults_sha256=file_digest(a.defaults), visual_review_sha256=file_digest(a.visual_review), tray_floor_probe_sha256=file_digest(a.tray_floor_proof),
                     native_scene_sha256=file_digest(scene_path), source_customs_sha256=file_digest(source), reservation_sha256=file_digest(a.reservations),
                     support_adaptation='Existing globally reserved table at exact measured native scale and height', production_released=False)
        if a.support_scene_proof:
            proof['support_scene_proof'] = dict(path=str(a.support_scene_proof), sha256=file_digest(a.support_scene_proof))
            proof['support_adaptation'] = 'Globally reserved default-scale table inserted into a collision-checked private scene; original objects unchanged'
        entry['campaign_provenance'] = proof
        template = Template('held_model__' + cls + '__initial_count_' + str(len(distractors)), 'disembedding', cls, 'eval_model', scene, room, tuple(refs),
                            digest(dict(entry=entry, bddl_sha256=file_digest(bddl))), {**proof, 'task_rooms': [room], 'inst_to_name': names,
                            'runtime_bddl_sha256': file_digest(bddl), 'customs_alias': alias, 'fresh_geometry': 'pending', 'fresh_demo': 'pending',
                            'fresh_replay': 'pending', 'coverage': 'Native area profile required before count expansion', 'perturbations': False})
        template = bind_variant(template, pool_path, {choice.object.instance: choice.candidate_id for choice in choices})
        check_reservations(template, policy, hashes); customs = directory / 'candidate_customs.json'; write_json_atomic(customs, {alias: entry})
        gp = GroundedProblem.load(alias, str(customs), str(directory / 'bddl'))
        native = list(iter_template_choices(gp, scene, np.random.default_rng(20261110), max_templates=1))
        require(len(native) == 1 and [(row.category, row.model) for row in native[0].groups['distractors']] == [(row['category'], row['model']) for row in distractors], 'native distractor choice differs')
        templates.append({**asdict(template), 'fingerprint': template.fingerprint})
        jobs.append(dict(candidate_id=template.template_id, class_id=cls, task='disembedding', alias=alias, scene=scene, room=room, split='eval_model',
                         template_fingerprint=template.fingerprint, source_manifest=str((a.out / 'validation_jobs.json').resolve()), required_new_grasp_key=gkey,
                         environment=scene_environment,
                         production_released=False, sample_argv=['python', '-m', 'ispatialgym.sampling.sample_isg', '--variant', alias, '--customs', str(customs),
                         '--bddl-root', str(directory / 'bddl'), '--scene', scene, '--templates', '1', '--instances-per-template', '1', '--max-attempts', '3', '--seed', '20261110',
                         '--no-write-back', '--out', '${CAMPAIGN_OUT}/validation/' + cls]))
        grasp_jobs.append(dict(id=gkey, legacy_key=gkey, category=target['category'], model=target['model'], scale=target['default_scale'], dataset=target['dataset'],
                              constraint='tabletop', rest_pose='canonical', rest_hint_wxyz=None, kind='tabletop', ready=True, readiness_source=target['readiness_source'],
                              asset_sha256=target['asset_sha256'], campaign_classes=[cls], source='campaign_held_disembedding_native_defaults'))
        requests.append(dict(class_id=cls, target_grasp_key=gkey, source='tray/gsxbym', destination='tray/mhhoga', target_scale=target['default_scale'],
                             tray_floor_proof=str(a.tray_floor_proof), tray_floor_proof_sha256=file_digest(a.tray_floor_proof),
                             height_keys=[trays['objects'][model]['map_height_key'] for model in ('gsxbym', 'mhhoga')],
                             status='fresh positive library and exact tray-floor maps required'))
    for filename, value in [('contract_candidates.json', dict(templates=templates)), ('validation_jobs.json', dict(jobs=jobs)),
                            ('tabletop_jobs.json', grasp_jobs), ('map_dependency_requests.json', requests)]: write_json_atomic(a.out / filename, value)
    print(json.dumps(dict(candidates=len(jobs), manifest_sha256=file_digest(a.out / 'validation_jobs.json'))))


if __name__ == '__main__':
    main()
