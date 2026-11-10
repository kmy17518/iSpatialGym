#!/usr/bin/env python3
"""Freeze calculator axial alignment and candle attachment category candidates."""
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
    for name in ('inventory', 'defaults', 'visual-review', 'support-scene-probe', 'reservations', 'reserved-asset-hashes', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args(); require(not a.out.exists(), 'new immutable output required')
    policy, hashes, doc, review, support_proof = [json.loads(path.read_text()) for path in
        (a.reservations, a.reserved_asset_hashes, a.defaults, a.visual_review, a.support_scene_probe)]
    require(hashes['reservation_sha256'] == sha(a.reservations), 'reservation hash companion differs')
    require(doc['status'] == 'native_defaults_and_geometry_recorded' and review['default_geometry_sha256'] == sha(a.defaults), 'native measurement/review differs')
    require(support_proof['status'] == 'private_support_geometry_and_reload_passed', 'private table proof incomplete')
    scene_path = Path(support_proof['native_scene_path']); require(sha(scene_path) == support_proof['native_scene_sha256'], 'private scene changed')
    scene, room = support_proof['scene'], support_proof['room']; init = json.loads(scene_path.read_text())['objects_info']['init_info']
    table = init[support_proof['object_name']]['args']; require(table['scale'] == support_proof['default_scale'], 'table scale proof differs')
    floor = next(x['args'] for x in init.values() if x['args'].get('category') == 'floors' and room in x['args'].get('in_rooms', []))
    measured = {x['category'] + '/' + x['model']: x for x in doc['objects']}
    ready = ReadinessCatalog(B1K / 'bddl3/bddl/generated_data'); scene_rows = {}
    for role, obj in [('support', table), ('floor', floor)]:
        status = ready.status(obj['category'], obj['model']); require(status['ready'], 'unready scene role')
        asset = B1K / 'datasets/behavior-1k-assets/objects' / obj['category'] / obj['model'] / 'usd' / (obj['model'] + '.encrypted.usd')
        scene_rows[role] = dict(category=obj['category'], model=obj['model'], default_scale=obj['scale'], asset_sha256=sha(asset),
            ready=True, readiness_source=status['readiness_source'], scene_object_name=obj['name'])
    source = a.inventory / 'frozen_customs_f24ea424.json'; originals = json.loads(source.read_text())
    records = {x['key']: x for x in json.loads((a.inventory / 'cluster_templates_with_scales.json').read_text())['records']}
    definitions, jobs, templates, dependencies, grasps = {}, [], [], [], []; a.out.mkdir(parents=True)
    for task, source_class, target_key, partner_key in [
        ('alignment-axial', 'alignment-axial_01', 'calculator/kwmmty', None),
        ('configuration_matching-attachment', 'configuration_matching-attachment_03', 'dip_candle/poexjo', 'candle_holder/wiufnv')]:
        roles = {**scene_rows, 'target': measured[target_key]}
        if partner_key:
            roles['partner'] = measured[partner_key]
        for role, row in roles.items():
            key = row['category'] + '/' + row['model']; require(row['ready'], 'unready category role')
            if role != 'floor':
                require(row['category'] in policy['category_holdout_categories'] and key in policy['category_holdout_models'], 'unreserved category role')
                require(hashes['ready_assets'][key]['asset_sha256'] == row['asset_sha256'], 'category role hash differs')
            if role in ('target', 'partner'):
                require(review['objects'][key]['status'] == 'passed' and sha(Path(row['asset_path'])) == row['asset_sha256'], 'native model proof changed')
        target = roles['target']; entry = copy.deepcopy(originals[records[source_class]['row']['alias']]); iddl = parse_file(str(ROOT / 'ispatialgym/assets/iddl' / entry['iddl']))
        for role in ('target', 'partner') if partner_key else ('target',):
            row = roles[role]; key = row['category'] + '/' + row['model']
            entry['roles'][role].update(category=row['category'], representative=row['model'], dataset=row['dataset'], scale={key: row['default_scale']})
            for name in ('grasp_keys', 'orientation_equivalences', 'rest_pose', 'rest_hint_wxyz'):
                entry['roles'][role].pop(name, None)
        entry['roles']['support']['category'] = table['category']
        gkey = entry_id(target['category'], target['model'], target['default_scale'], 'attachment_free') if partner_key else None
        if gkey:
            entry['roles']['target']['grasp_keys'] = {target_key: gkey}
        entry['scenes'] = {scene: dict(rooms=[room], bindings={'floor': {'room': room}, 'support': [table['name']],
            'support_surfaces': [dict(name=table['name'], init='ontop', top_z=float(support_proof['actual_height_key']), row='campaign_category_table_tasks')]},
            whitelist={role: {row['category']: [row['model']]} for role, row in roles.items() if role in ('target', 'partner')}, blacklist={})}
        entry['constraints'] = []
        for name in list(entry):
            if name.startswith('fallback') or name == 'superseded_by':
                entry.pop(name)
        for name in list(entry['generation']):
            if name.startswith('_') or name == 'remove_objects':
                entry['generation'].pop(name)
        entry['generation'].update(templates_max=1, instances_per_template=1)
        entry['roles'] = {name: entry['roles'][name] for name in iddl.roles}
        alias = iddl.variant_template.format(target=target['category'], partner=roles.get('partner', {}).get('category', ''), support=table['category'])
        cls = task + '-' + target['category'] + ('-' + roles['partner']['category'] if partner_key else '')
        did = 29900 + len(definitions); entry['definition_id'] = did; grounds = {}
        for role in iddl.roles:
            if role == 'floor':
                grounds[role] = RoleGrounding(role, 'scene', same_as='floor')
            else:
                category = roles[role]['category']; grounds[role] = RoleGrounding(role, 'sampleable' if entry['roles'][role].get('sampleable') else 'scene',
                    category=category, synset=synset_of_category(category))
        activity = task.split('-')[0]; rendered = render(iddl, grounds, did, alias, activity)
        bddl = a.out / 'bddl' / activity / f'problem{did}.bddl'; bddl.parent.mkdir(parents=True, exist_ok=True); bddl.write_text(rendered.text)
        refs, names = [], {}
        for role, row in roles.items():
            instance = rendered.instances[role]
            refs.append(ObjectRef(instance=instance, role=role if role in ('target', 'partner') else 'support', category=row['category'], model=row['model'],
                asset_sha256=row['asset_sha256'], default_scale=tuple(row['default_scale']), semantic_kind='floor' if role == 'floor' else 'object',
                ready=True, readiness_source=row['readiness_source']))
            if row.get('scene_object_name'):
                names[instance] = row['scene_object_name']
        proof = dict(defaults_sha256=sha(a.defaults), visual_review_sha256=sha(a.visual_review), support_scene_probe_sha256=sha(a.support_scene_probe),
            native_scene_sha256=sha(scene_path), source_customs_sha256=sha(source), source_class_id=source_class, reservation_sha256=sha(a.reservations), production_released=False)
        entry['campaign_provenance'] = proof; definitions[alias] = entry
        template = Template('held_category__' + cls, task, cls, 'eval_category', scene, room, tuple(refs), digest(dict(entry=entry, bddl_sha256=sha(bddl))),
            {**proof, 'task_rooms': [room], 'inst_to_name': names, 'runtime_bddl_sha256': sha(bddl), 'customs_alias': alias,
             'fresh_geometry': 'pending', 'fresh_demo': 'pending', 'fresh_replay': 'pending', 'perturbations': False})
        check_reservations(template, policy, hashes); templates.append({**asdict(template), 'fingerprint': template.fingerprint})
        jobs.append(dict(candidate_id=template.template_id, class_id=cls, task=task, alias=alias, scene=scene, room=room, split='eval_category',
            template_fingerprint=template.fingerprint, environment={'ISPATIALGYM_STABLE_SCENE_DIR': str(scene_path.parent)}, source_manifest=str((a.out / 'validation_jobs.json').resolve()),
            required_new_grasp_key=gkey, production_released=False, sample_argv=['python', '-m', 'ispatialgym.sampling.sample_isg', '--variant', alias,
            '--customs', str((a.out / 'candidate_customs.json').resolve()), '--bddl-root', str((a.out / 'bddl').resolve()), '--scene', scene, '--templates', '1',
            '--instances-per-template', '1', '--max-attempts', '3', '--seed', '20261110', '--no-write-back', '--out', '${CAMPAIGN_OUT}/validation/' + cls]))
        request = dict(class_id=cls, category=target['category'], model=target['model'], scale=target['default_scale'], asset_sha256=target['asset_sha256'],
            height_key=support_proof['actual_height_key'], ready=True, readiness_source=target['readiness_source'], status='native_maps_and_task_geometry_pending')
        if partner_key:
            pair_metadata = {}
            for role in ('target', 'partner'):
                row = roles[role]; path = B1K / 'datasets/behavior-1k-assets/objects' / row['category'] / row['model'] / 'misc/metadata.json'
                links = json.loads(path.read_text()).get('meta_links', {})
                pair_metadata[role] = dict(path=str(path), sha256=sha(path), attachment_links={name: info['attachment'] for name, info in links.items() if info.get('attachment')})
            require(any('dipcandleM' in row for row in pair_metadata['target']['attachment_links'].values()) and
                    any('dipcandleF' in row for row in pair_metadata['partner']['attachment_links'].values()), 'matching native candle attachment metalinks absent')
            request.update(pipeline='attachment', grasp_key=gkey, partner=partner_key, partner_scale=roles['partner']['default_scale'], native_metalink_evidence=pair_metadata)
            grasps.append(dict(id=gkey, legacy_key=gkey, category=target['category'], model=target['model'], scale=target['default_scale'], dataset=target['dataset'],
                constraint='attachment_free', rest_pose='canonical', rest_hint_wxyz=None, kind='tabletop', asset_sha256=target['asset_sha256'], ready=True,
                readiness_source=target['readiness_source'], campaign_classes=[cls], source='campaign_category_attachment'))
        else:
            request.update(pipeline='contact', kinds=[{'kind': 'push_blade', 'arm': 'right'}, {'kind': 'axial_brace', 'arm': 'left'}],
                native_rest='must be measured for this model; no source-model contact/rest metadata transfers')
        dependencies.append(request)
    custom = a.out / 'candidate_customs.json'; custom.write_text(json.dumps(definitions, indent=2) + '\n')
    for job in jobs:
        problem = GroundedProblem.load(job['alias'], str(custom), str(a.out / 'bddl')); require(problem.rooms(scene) == [room], 'native category room mismatch')
    for filename, value in [('contract_candidates.json', {'templates': templates}), ('validation_jobs.json', {'jobs': jobs}),
                            ('tabletop_jobs.json', grasps), ('native_dependency_requests.json', dependencies)]:
        (a.out / filename).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(candidates=len(jobs), manifest_sha256=sha(a.out / 'validation_jobs.json'))))


if __name__ == '__main__':
    main()
