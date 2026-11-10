#!/usr/bin/env python3
"""Compile heldout wardrobe articulation after private placement/reload review."""
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
from ispatialgym.sampling.bddl_from_iddl import RoleGrounding, render
from ispatialgym.sampling.iddl import parse_file
from ispatialgym.sampling.knowledge import synset_of_category
from ispatialgym.sampling.v2.problem import GroundedProblem


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('inventory', 'defaults', 'scene-probe', 'visual-review', 'reservations', 'reserved-asset-hashes', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args(); require(not a.out.exists(), 'new immutable output required')
    policy, hashes, defaults, probe, review = [json.loads(path.read_text()) for path in
        (a.reservations, a.reserved_asset_hashes, a.defaults, a.scene_probe, a.visual_review)]
    require(hashes['reservation_sha256'] == sha(a.reservations), 'reservation hash companion differs')
    require(defaults['status'] == 'native_defaults_and_geometry_recorded', 'native defaults incomplete')
    require(probe['status'] == 'private_context_geometry_and_reload_passed' and review['status'] == 'passed'
        and review['context_probe_sha256'] == sha(a.scene_probe), 'private wardrobe scene review incomplete')
    row = next(x for x in defaults['objects'] if x['category'] == 'wardrobe' and x['model'] == 'brimns')
    measured = next(x for x in probe['objects'] if x['model_key'] == 'wardrobe/brimns')
    require(row['ready'] and row['asset_sha256'] == measured['asset_sha256'] == hashes['ready_assets']['wardrobe/brimns']['asset_sha256'], 'wardrobe identity/readiness differs')
    require(sha(Path(row['asset_path'])) == row['asset_sha256'] and row['default_scale'] == measured['default_scale'], 'wardrobe asset/scale changed')
    require(measured.get('joint_sweeps') and all(not x['blockers'] for x in measured['joint_sweeps']), 'native environmental joint sweep proof missing')
    scene_path = Path(probe['native_scene_path']); require(sha(scene_path) == probe['native_scene_sha256'], 'private scene changed')
    scene, room, name = probe['scene'], probe['room'], measured['name']; scene_doc = json.loads(scene_path.read_text()); init = scene_doc['objects_info']['init_info']
    native = init[name]['args']; require(native['scale'] == row['default_scale'] and native['category'] == 'wardrobe' and native['model'] == 'brimns', 'native scene target differs')
    floor = next(x['args'] for x in init.values() if x['args'].get('category') == 'floors' and room in x['args'].get('in_rooms', []))
    ready = ReadinessCatalog(B1K / 'bddl3/bddl/generated_data'); fs = ready.status(floor['category'], floor['model']); require(fs['ready'], 'unready floor')
    floor_usd = B1K / 'datasets/behavior-1k-assets/objects' / floor['category'] / floor['model'] / 'usd' / (floor['model'] + '.encrypted.usd')
    source = a.inventory / 'frozen_customs_f24ea424.json'; originals = json.loads(source.read_text())
    records = {x['key']: x for x in json.loads((a.inventory / 'cluster_templates_with_scales.json').read_text())['records']}
    jobs, templates, dependencies = [], [], []; a.out.mkdir(parents=True)
    for direction in ('open', 'close'):
        task = 'configuration_matching-articulation_' + direction + '_large_scale'; cls = task + '-wardrobe'; source_class = task + '_01'
        for joint in ('j_leftdoor', 'j_rightdoor'):
            require(joint in row['joints'] and row['joints'][joint]['upper_limit'] > row['joints'][joint]['lower_limit'], 'native wardrobe joint missing')
            require({x['fraction'] for x in measured['joint_sweeps'] if x['joint'] == joint} == {0., .25, .5, .75, 1.}, 'incomplete native joint sweep')
            entry = copy.deepcopy(originals[records[source_class]['row']['alias']]); iddl = parse_file(str(ROOT / 'ispatialgym/assets/iddl' / entry['iddl']))
            entry['roles']['target'].update(category='wardrobe', joint={scene + '/' + name: joint}, handle_key={scene + '/' + name: 'wardrobe__brimns'})
            entry['roles']['target'].pop('handle_row', None)
            entry['scenes'] = {scene: dict(rooms=[room], bindings={'floor': {'room': room}, 'target': [name]}, whitelist={}, blacklist={})}
            entry['constraints'] = []
            for key in list(entry):
                if key.startswith('fallback') or key == 'superseded_by':
                    entry.pop(key)
            for key in list(entry['generation']):
                if key.startswith('_') or key == 'remove_objects':
                    entry['generation'].pop(key)
            entry['generation'].update(templates_max=1, instances_per_template=1)
            entry['roles'] = {key: entry['roles'][key] for key in iddl.roles}
            alias = iddl.variant_template.format(target='wardrobe'); did = 30200 + len(jobs); entry['definition_id'] = did
            rendered = render(iddl, {'target': RoleGrounding('target', 'scene', category='wardrobe', synset=synset_of_category('wardrobe')),
                'floor': RoleGrounding('floor', 'scene', same_as='floor')}, did, alias, 'configuration_matching')
            directory = a.out / (direction + '_' + joint); directory.mkdir(); bddl = directory / 'bddl/configuration_matching' / f'problem{did}.bddl'
            bddl.parent.mkdir(parents=True); bddl.write_text(rendered.text)
            refs = (ObjectRef(instance=rendered.instances['target'], role='target', category='wardrobe', model='brimns', asset_sha256=row['asset_sha256'],
                        default_scale=tuple(row['default_scale']), joint=joint, ready=True, readiness_source=row['readiness_source']),
                    ObjectRef(instance=rendered.instances['floor'], role='support', category=floor['category'], model=floor['model'], asset_sha256=sha(floor_usd),
                        default_scale=tuple(floor['scale']), semantic_kind='floor', ready=True, readiness_source=fs['readiness_source']))
            proof = dict(defaults_sha256=sha(a.defaults), scene_probe_sha256=sha(a.scene_probe), visual_review_sha256=sha(a.visual_review),
                native_scene_sha256=sha(scene_path), reservation_sha256=sha(a.reservations), source_customs_sha256=sha(source), source_class_id=source_class, production_released=False)
            entry['campaign_provenance'] = proof
            template = Template('held_category__wardrobe_' + direction + '_' + joint, task, cls, 'eval_category', scene, room, refs,
                digest(dict(entry=entry, bddl_sha256=sha(bddl))), {**proof, 'task_rooms': [room], 'inst_to_name': {rendered.instances['target']: name, rendered.instances['floor']: floor['name']},
                    'runtime_bddl_sha256': sha(bddl), 'customs_alias': alias, 'native_joint': row['joints'][joint], 'fresh_geometry': 'pending', 'fresh_demo': 'pending', 'fresh_replay': 'pending', 'perturbations': False})
            check_reservations(template, policy, hashes); templates.append({**asdict(template), 'fingerprint': template.fingerprint})
            customs = directory / 'candidate_customs.json'; customs.write_text(json.dumps({alias: entry}, indent=2) + '\n')
            gp = GroundedProblem.load(alias, str(customs), str(directory / 'bddl')); require(gp.rooms(scene) == [room], 'native category articulation room mismatch')
            jobs.append(dict(candidate_id=template.template_id, class_id=cls, task=task, alias=alias, scene=scene, room=room, split='eval_category', template_fingerprint=template.fingerprint,
                environment={'ISPATIALGYM_STABLE_SCENE_DIR': str(scene_path.parent)}, source_manifest=str((a.out / 'validation_jobs.json').resolve()), production_released=False,
                sample_argv=['python', '-m', 'ispatialgym.sampling.sample_isg', '--variant', alias, '--customs', str(customs.resolve()), '--bddl-root', str((directory / 'bddl').resolve()),
                    '--scene', scene, '--templates', '1', '--instances-per-template', '1', '--max-attempts', '3', '--seed', '20261110', '--no-write-back', '--out', '${CAMPAIGN_OUT}/validation/' + template.template_id]))
            dependencies.append(dict(candidate_id=template.template_id, class_id=cls, pipeline='scene_handle_kinematics', category='wardrobe', model='brimns',
                scale=row['default_scale'], asset_sha256=row['asset_sha256'], joint=joint, native_joint=row['joints'][joint], handle_key='wardrobe__brimns',
                scene=scene, room=room, scene_object=name, scene_root_state=scene_doc['state']['registry']['object_registry'][name]['root_link'],
                direction=direction, generation=entry['generation'], status='native_handle_and_task_geometry_pending'))
    for filename, value in [('validation_jobs.json', {'jobs': jobs}), ('contract_candidates.json', {'templates': templates}), ('articulation_dependency_requests.json', dependencies)]:
        (a.out / filename).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(candidates=len(jobs), manifest_sha256=sha(a.out / 'validation_jobs.json'))))


if __name__ == '__main__':
    main()
