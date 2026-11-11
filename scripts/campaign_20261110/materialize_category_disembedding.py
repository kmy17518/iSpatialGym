#!/usr/bin/env python3
"""Compile an exact initial category disembedding variant and maximum pool.

The admitted pool allows later finite count variants. Its capacity is a bound,
not a packing claim; coverage and physical fit must come from native probes.
"""
import argparse
import copy
from dataclasses import asdict, replace
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
from scripts.campaign_20261110.binding_pools import BindingPool, PoolCandidate, bind_variant, write_pool
from scripts.campaign_20261110.group_lane import check_reservations
from scripts.campaign_20261110.readiness_catalog import ReadinessCatalog


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('inventory', 'defaults', 'visual-review', 'tray-defaults', 'tray-visual-review', 'support-scene-probe',
                 'reservations', 'reserved-asset-hashes', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--snapshot', type=Path, default=ROOT)
    p.add_argument('--destination-side', choices=('left', 'right'))
    a = p.parse_args()
    snapshot = a.snapshot.resolve(); b1k = snapshot / 'Behavior-1K'
    sys.path[:0] = [str(snapshot), str(b1k / 'bddl3')]
    from ispatialgym.grasping.schema import entry_id
    from ispatialgym.sampling.bddl_from_iddl import RoleGrounding, render
    from ispatialgym.sampling.bddl_writer import Bindings
    from ispatialgym.sampling.iddl import parse_file
    from ispatialgym.sampling.knowledge import synset_of_category
    from ispatialgym.sampling.v2.problem import GroundedProblem, iter_template_choices
    require(not a.out.exists(), 'new immutable output required')
    policy, hashes = [json.loads(path.read_text()) for path in (a.reservations, a.reserved_asset_hashes)]
    require(hashes['reservation_sha256'] == sha(a.reservations), 'reservation hash companion differs')
    measured, provenance = {}, []
    for defaults_path, review_path in [(a.defaults, a.visual_review), (a.tray_defaults, a.tray_visual_review)]:
        doc, review = [json.loads(path.read_text()) for path in (defaults_path, review_path)]
        require(doc['status'] == 'native_defaults_and_geometry_recorded' and review['default_geometry_sha256'] == sha(defaults_path), 'native defaults/visual review mismatch')
        for row in doc['objects']:
            key = row['category'] + '/' + row['model']
            if key in ('rubber_eraser/cungod', 'crayon/coapeh', 'crayon/csglmn', 'tray/hjxczh', 'tray/thkphg'):
                require(row['ready'] and review['objects'][key]['status'] == 'passed', 'unreviewed or unready category role')
                require(key in policy['category_holdout_models'] and hashes['ready_assets'][key]['asset_sha256'] == row['asset_sha256'], 'category role not currently reserved')
                require(sha(Path(row['asset_path'])) == row['asset_sha256'], 'measured role USD bytes changed')
                measured[key] = row
        provenance.append(dict(defaults=str(defaults_path), defaults_sha256=sha(defaults_path), review=str(review_path), review_sha256=sha(review_path),
            measurement_reservation_sha256=doc['reservation_sha256']))
    require(len(measured) == 5, 'category disembedding role evidence incomplete')
    support_proof = json.loads(a.support_scene_probe.read_text())
    require(support_proof['status'] == 'private_support_geometry_and_reload_passed', 'private table evidence incomplete')
    scene_path = Path(support_proof['native_scene_path']); require(sha(scene_path) == support_proof['native_scene_sha256'], 'private support scene changed')
    scene, room = support_proof['scene'], support_proof['room']
    init = json.loads(scene_path.read_text())['objects_info']['init_info']
    support = init[support_proof['object_name']]['args']
    require(support['scale'] == support_proof['default_scale'], 'private table scale mismatch')
    floor = next(x['args'] for x in init.values() if x['args'].get('category') == 'floors' and room in x['args'].get('in_rooms', []))
    ready = ReadinessCatalog(b1k / 'bddl3/bddl/generated_data')
    role_rows = {'target': measured['rubber_eraser/cungod'], 'distractors': measured['crayon/coapeh'],
        'source': measured['tray/hjxczh'], 'destination': measured['tray/thkphg']}
    for role, obj in [('floor', floor), ('support', support)]:
        status = ready.status(obj['category'], obj['model']); require(status['ready'], 'unready scene role')
        asset = b1k / 'datasets/behavior-1k-assets/objects' / obj['category'] / obj['model'] / 'usd' / (obj['model'] + '.encrypted.usd')
        role_rows[role] = dict(category=obj['category'], model=obj['model'], default_scale=obj['scale'], asset_sha256=sha(asset),
            ready=True, readiness_source=status['readiness_source'], scene_object_name=obj['name'])
    source = a.inventory / 'frozen_customs_f24ea424.json'; originals = json.loads(source.read_text())
    record = next(x for x in json.loads((a.inventory / 'cluster_templates_with_scales.json').read_text())['records'] if x['key'] == 'disembedding_02')
    entry = copy.deepcopy(originals[record['row']['alias']]); source_definition = digest(entry)
    original_side = entry['generation'].get('destination_side')
    if a.destination_side:
        entry['generation']['destination_side'] = a.destination_side
    iddl = parse_file(str(snapshot / 'ispatialgym/assets/iddl' / entry['iddl']))
    for role in ('target', 'source', 'destination', 'distractors'):
        row = role_rows[role]; key = row['category'] + '/' + row['model']
        entry['roles'][role].update(category=[row['category']] if role == 'distractors' else row['category'],
            representative=row['model'], dataset=row['dataset'], scale={key: row['default_scale']})
        for name in ('grasp_keys', 'orientation_equivalences', 'rest_pose', 'rest_hint_wxyz'):
            entry['roles'][role].pop(name, None)
    entry['roles']['support']['category'] = support['category']
    target = role_rows['target']; gkey = entry_id(target['category'], target['model'], target['default_scale'], 'tabletop')
    entry['roles']['target']['grasp_keys'] = {'rubber_eraser/cungod': gkey}
    entry['generation']['groups']['distractors'] = {'size': 3, 'draw': 'same_model', 'members': {'crayon': 3}}
    entry['generation']['distractor_budget'] = {'budget_fraction': .85, 'n_distractors': 3, 'source': 'initial exact native measurement candidate; coverage unmeasured'}
    for name in list(entry['generation']):
        if name.startswith('_') or name in ('remove_objects', 'fixed_layout'):
            entry['generation'].pop(name)
    for name in list(entry):
        if name.startswith('fallback') or name == 'superseded_by':
            entry.pop(name)
    entry['generation'].update(templates_max=1, instances_per_template=1)
    entry['constraints'] = []
    whitelist = {role: {role_rows[role]['category']: [role_rows[role]['model']]} for role in ('target', 'source', 'destination', 'distractors')}
    entry['scenes'] = {scene: dict(rooms=[room], bindings={'floor': {'room': room}, 'support': [support['name']],
        'support_surfaces': [dict(name=support['name'], init='ontop', top_z=float(support_proof['actual_height_key']), row='campaign_category_disembedding')]},
        whitelist=whitelist, blacklist={})}
    entry['roles'] = {name: entry['roles'][name] for name in iddl.roles}
    alias = iddl.variant_template.format(target='rubber_eraser', distractors='crayon_3', support=support['category'])
    did = 29800; entry['definition_id'] = did; grounds = {}
    for role in iddl.roles:
        if role == 'floor':
            grounds[role] = RoleGrounding(role, 'scene', same_as='floor')
        elif role == 'distractors':
            grounds[role] = RoleGrounding(role, 'group', members=[('crayon', synset_of_category('crayon'))] * 3)
        else:
            category = role_rows[role]['category']; grounds[role] = RoleGrounding(role, 'sampleable' if entry['roles'][role].get('sampleable') else 'scene',
                category=category, synset=synset_of_category(category))
    rendered = render(iddl, grounds, did, alias, 'disembedding'); a.out.mkdir(parents=True)
    bddl = a.out / 'bddl/disembedding/problem29800.bddl'; bddl.parent.mkdir(parents=True); bddl.write_text(rendered.text)
    refs, names = [], {}
    for role, row in role_rows.items():
        instances = rendered.instances['distractors.members'] if role == 'distractors' else [rendered.instances[role]]
        for instance in instances:
            refs.append(ObjectRef(instance=instance, role='target' if role == 'target' else 'distractor' if role == 'distractors' else 'support',
                category=row['category'], model=row['model'], asset_sha256=row['asset_sha256'], default_scale=tuple(row['default_scale']),
                semantic_kind='floor' if role == 'floor' else 'tray' if role in ('source', 'destination') else 'object', ready=True, readiness_source=row['readiness_source']))
            if row.get('scene_object_name'):
                names[instance] = row['scene_object_name']
    target_ref = next(x for x in refs if x.role == 'target'); distractor_ref = next(x for x in refs if x.role == 'distractor')
    second = measured['crayon/csglmn']; alternative = replace(distractor_ref, instance='pool_crayon_csglmn', model=second['model'], asset_sha256=second['asset_sha256'],
        default_scale=tuple(second['default_scale']), readiness_source=second['readiness_source'])
    choices = (PoolCandidate('eraser_cungod', target_ref), PoolCandidate('crayon_coapeh', distractor_ref, 40), PoolCandidate('crayon_csglmn', alternative, 40))
    cls = 'disembedding-rubber_eraser-crayon-pedestal_table'
    pool = BindingPool('category_disembedding_stationery_pool_v1', cls, 'eval_category', scene, room, tuple(x for x in refs if x.role == 'support'), choices,
        {'target': (1, 1), 'distractor': (1, 40)}, source_definition, task_rooms=(room,), provenance={'native_evidence': provenance,
            'support_scene_probe_sha256': sha(a.support_scene_probe), 'capacity_note': '40 is a finite admission cap, not a claim of physical fit or crowdedness coverage'})
    pool_path = write_pool(a.out / 'binding_pool.json', pool)
    proof = dict(native_evidence=provenance, support_scene_probe_sha256=sha(a.support_scene_probe), native_scene_sha256=sha(scene_path),
        source_customs_sha256=sha(source), reservation_sha256=sha(a.reservations), production_released=False,
        runtime_role_order=list(entry['roles']))
    if a.destination_side:
        proof['tray_side_adaptation'] = dict(original_destination_side=original_side, destination_side=a.destination_side,
            reason='Fresh permitted left/right tray layout; native geometry, demo and replay checks unchanged')
    entry['campaign_provenance'] = proof
    candidate_id = 'held_category__stationery_count3' + ('__destination_' + a.destination_side if a.destination_side else '')
    template = Template(candidate_id, 'disembedding', cls, 'eval_category', scene, room, tuple(refs), digest(dict(entry=entry, bddl_sha256=sha(bddl))),
        {**proof, 'task_rooms': [room], 'inst_to_name': names, 'runtime_bddl_sha256': sha(bddl), 'customs_alias': alias,
         'fresh_geometry': 'pending', 'fresh_demo': 'pending', 'fresh_replay': 'pending', 'coverage': 'requires native profile before additional count variants', 'perturbations': False})
    template = bind_variant(template, pool_path, {x.instance: 'eraser_cungod' if x.role == 'target' else 'crayon_coapeh' for x in refs if x.role != 'support'})
    check_reservations(template, policy, hashes)
    custom = a.out / 'candidate_customs.json'; custom.write_text(json.dumps({alias: entry}, indent=2) + '\n')
    native = GroundedProblem.load(alias, str(custom), str(a.out / 'bddl')); require(native.rooms(scene) == [room], 'native category room mismatch')
    import numpy as np
    choices = list(iter_template_choices(native, scene, np.random.default_rng(20261110), max_templates=1))
    require(len(choices) == 1 and [(x.category, x.model) for x in choices[0].groups['distractors']] == [('crayon', 'coapeh')] * 3,
        'native repeated-model category enumeration differs')
    bindings = Bindings()
    for role in dict.fromkeys(['floor', *native.scene_roles()]):
        actual = bindings.add(role, synset_of_category(role_rows[role]['category']))
        require(actual.name == rendered.instances[role], 'native scene-role BDDL allocation differs: ' + role)
    for role in native.sampleable_roles():
        actual = bindings.add(role, choices[0].models[role].synset)
        require(actual.name == rendered.instances[role], 'native sampleable-role BDDL allocation differs: ' + role)
    for role in native.group_roles():
        actual = [bindings.add(role, member.synset).name for member in choices[0].groups[role]]
        require(actual == rendered.instances[role + '.members'], 'native group-role BDDL allocation differs: ' + role)
    (a.out / 'native_binding_order_check.json').write_text(json.dumps(dict(runtime_role_order=list(native.roles),
        customs_sha256=sha(custom), bddl_sha256=sha(bddl), bindings=[asdict(row) for row in bindings.all()],
        native_allocation_matches_rendered_bddl=True), indent=2) + '\n')
    job = dict(candidate_id=template.template_id, class_id=cls, task='disembedding', alias=alias, scene=scene, room=room, split='eval_category',
        template_fingerprint=template.fingerprint, source_manifest=str((a.out / 'validation_jobs.json').resolve()), required_new_grasp_key=gkey,
        environment={'ISPATIALGYM_STABLE_SCENE_DIR': str(scene_path.parent)}, production_released=False, required_tray_floor_height_key='native measurement pending',
        sample_argv=['python', '-m', 'ispatialgym.sampling.sample_isg', '--variant', alias, '--customs', str(custom.resolve()), '--bddl-root', str((a.out / 'bddl').resolve()),
        '--scene', scene, '--templates', '1', '--instances-per-template', '1', '--max-attempts', '3', '--seed', '20261110', '--no-write-back', '--out', '${CAMPAIGN_OUT}/validation/' + cls])
    dependency = dict(class_id=cls, target_grasp_key=gkey, source_tray='tray/hjxczh', destination_tray='tray/thkphg',
        placed_key='tray/hjxczh@focus_centred_edge_aligned', target_scale=target['default_scale'], source_scale=measured['tray/hjxczh']['default_scale'],
        height_key='read exact category_disembedding_tray_floors_v1 report', target_library_reuse='same exact tabletop key as category small pairs; positive library validation required')
    for filename, value in [('contract_candidates.json', {'templates': [{**asdict(template), 'fingerprint': template.fingerprint}]}),
                            ('validation_jobs.json', {'jobs': [job]}), ('map_dependency_requests.json', [dependency])]:
        (a.out / filename).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(candidates=1, pool_sha256=pool.fingerprint, manifest_sha256=sha(a.out / 'validation_jobs.json'))))


if __name__ == '__main__':
    main()
