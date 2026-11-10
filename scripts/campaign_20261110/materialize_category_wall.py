#!/usr/bin/env python3
"""Freeze reviewed poster alignment on three actual heldout-room walls."""
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
    for name in ('inventory', 'defaults', 'frame-review', 'support-scene-probe', 'reservations', 'reserved-asset-hashes', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args(); require(not a.out.exists(), 'new immutable output required')
    policy, hashes, defaults, review, context = [json.loads(path.read_text()) for path in
        (a.reservations, a.reserved_asset_hashes, a.defaults, a.frame_review, a.support_scene_probe)]
    require(hashes['reservation_sha256'] == sha(a.reservations), 'reservation hash companion changed')
    require(defaults['status'] == 'native_defaults_and_geometry_recorded' and review['status'] == 'passed'
        and review['evidence']['default_geometry_sha256'] == sha(a.defaults), 'reviewed poster frame proof incomplete')
    target = next(row for row in defaults['objects'] if row['category'] + '/' + row['model'] == review['model_key'])
    require(target['ready'] and target['asset_sha256'] == review['asset_sha256'] == hashes['ready_assets'][review['model_key']]['asset_sha256'], 'poster readiness/identity differs')
    require(sha(Path(target['asset_path'])) == target['asset_sha256'] and target['default_scale'] == review['default_scale'], 'poster asset/scale changed')
    require(sha(Path(review['evidence']['metadata_path'])) == review['evidence']['metadata_sha256'], 'poster rear-mount evidence changed')
    scene_path = Path(context['native_scene_path']); require(context['status'] == 'private_support_geometry_and_reload_passed' and sha(scene_path) == context['native_scene_sha256'], 'private context scene differs')
    scene, room = context['scene'], context['room']; init = json.loads(scene_path.read_text())['objects_info']['init_info']
    floor = next(x['args'] for x in init.values() if x['args'].get('category') == 'floors' and room in x['args'].get('in_rooms', []))
    source = a.inventory / 'frozen_customs_f24ea424.json'; originals = json.loads(source.read_text())
    record = next(x for x in json.loads((a.inventory / 'cluster_templates_with_scales.json').read_text())['records'] if x['key'] == 'alignment-world_vertical_01')
    ready = ReadinessCatalog(B1K / 'bddl3/bddl/generated_data'); jobs, templates, requests = [], [], []; a.out.mkdir(parents=True)
    task = 'alignment-world_vertical'; cls = task + '-poster'
    for index, wall_name in enumerate(('walls_bpehln_0', 'walls_wxbrqm_0', 'walls_vkjkbk_0')):
        wall = init[wall_name]['args']; require(wall['category'] == 'walls' and room in wall['in_rooms'], 'wall is outside heldout room')
        entry = copy.deepcopy(originals[record['row']['alias']]); iddl = parse_file(str(ROOT / 'ispatialgym/assets/iddl' / entry['iddl']))
        key = review['model_key']; entry['roles']['target'].update(category=target['category'], representative=target['model'], dataset=target['dataset'], scale={key: target['default_scale']})
        entry['roles']['target'].pop('orientation_equivalences', None)
        entry['generation']['front_axis'] = {key: review['front_axis']}; entry['generation']['up_axis'] = {key: review['up_axis']}
        entry['generation'].update(templates_max=1, instances_per_template=1)
        for name in list(entry['generation']):
            if name.startswith('_') or name == 'remove_objects':
                entry['generation'].pop(name)
        for name in list(entry):
            if name.startswith('fallback') or name == 'superseded_by':
                entry.pop(name)
        entry['constraints'] = []
        entry['scenes'] = {scene: dict(rooms=[room], bindings={'floor': {'room': room}, 'support': [wall_name]},
            whitelist={'target': {target['category']: [target['model']]}}, blacklist={})}
        entry['roles'] = {name: entry['roles'][name] for name in iddl.roles}
        alias = iddl.variant_template.format(target=target['category'], support='walls'); did = 30100 + index; entry['definition_id'] = did
        grounds = {role: RoleGrounding(role, 'sampleable' if role == 'target' else 'scene', category=category, synset=synset_of_category(category))
            for role, category in [('target', target['category']), ('support', 'walls')]}
        grounds['floor'] = RoleGrounding('floor', 'scene', same_as='floor')
        rendered = render(iddl, grounds, did, alias, 'alignment'); directory = a.out / wall_name; directory.mkdir()
        bddl = directory / 'bddl/alignment' / f'problem{did}.bddl'; bddl.parent.mkdir(parents=True); bddl.write_text(rendered.text)
        refs, names = [], {}
        for role, row in [('target', target), ('support', wall), ('floor', floor)]:
            category, model = row['category'], row['model']; status = ready.status(category, model); require(status['ready'], 'unready target/structural role')
            asset = B1K / 'datasets/behavior-1k-assets/objects' / category / model / 'usd' / (model + '.encrypted.usd')
            refs.append(ObjectRef(instance=rendered.instances[role], role='target' if role == 'target' else 'support', category=category, model=model,
                asset_sha256=sha(asset), default_scale=tuple(row['default_scale'] if role == 'target' else row['scale']),
                semantic_kind='object' if role == 'target' else 'wall' if role == 'support' else 'floor', ready=True, readiness_source=status['readiness_source']))
            if role != 'target':
                names[rendered.instances[role]] = row['name']
        proof = dict(defaults_sha256=sha(a.defaults), frame_review_sha256=sha(a.frame_review), source_customs_sha256=sha(source),
            native_scene_sha256=sha(scene_path), support_scene_probe_sha256=sha(a.support_scene_probe), reservation_sha256=sha(a.reservations), production_released=False)
        entry['campaign_provenance'] = proof
        template = Template('held_category__poster__' + wall_name, task, cls, 'eval_category', scene, room, tuple(refs), digest(dict(entry=entry, bddl_sha256=sha(bddl))),
            {**proof, 'task_rooms': [room], 'inst_to_name': names, 'runtime_bddl_sha256': sha(bddl), 'customs_alias': alias,
            'reviewed_frame': {'front_axis': review['front_axis'], 'up_axis': review['up_axis']}, 'fresh_geometry': 'pending', 'fresh_demo': 'pending', 'fresh_replay': 'pending', 'perturbations': False})
        check_reservations(template, policy, hashes); templates.append({**asdict(template), 'fingerprint': template.fingerprint})
        customs = directory / 'candidate_customs.json'; customs.write_text(json.dumps({alias: entry}, indent=2) + '\n')
        problem = GroundedProblem.load(alias, str(customs), str(directory / 'bddl')); require(problem.rooms(scene) == [room], 'native category wall room mismatch')
        jobs.append(dict(candidate_id=template.template_id, class_id=cls, task=task, alias=alias, scene=scene, room=room, split='eval_category', template_fingerprint=template.fingerprint,
            environment={'ISPATIALGYM_STABLE_SCENE_DIR': str(scene_path.parent)}, source_manifest=str((a.out / 'validation_jobs.json').resolve()), production_released=False,
            status='native_wall_spot_visibility_contact_geometry_pending', sample_argv=['python', '-m', 'ispatialgym.sampling.sample_isg', '--variant', alias, '--customs', str(customs.resolve()),
            '--bddl-root', str((directory / 'bddl').resolve()), '--scene', scene, '--templates', '1', '--instances-per-template', '1', '--max-attempts', '3', '--seed', '20261110',
            '--no-write-back', '--out', '${CAMPAIGN_OUT}/validation/' + template.template_id]))
        requests.append(dict(candidate_id=template.template_id, category=target['category'], model=target['model'], scale=target['default_scale'],
            native_frame_review=str(a.frame_review), native_frame_review_sha256=sha(a.frame_review), wall=wall_name, hang_height_m=entry['generation']['hang_height_m'],
            contact_maps='native wall-brace maps are optional acceleration; preserve existing native fallback and exact IK/visibility gates'))
    for filename, value in [('validation_jobs.json', {'jobs': jobs}), ('contract_candidates.json', {'templates': templates}), ('native_dependency_requests.json', requests)]:
        (a.out / filename).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(candidates=len(jobs), manifest_sha256=sha(a.out / 'validation_jobs.json'))))


if __name__ == '__main__':
    main()
