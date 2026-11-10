#!/usr/bin/env python3
"""Compile finite exact count variants from measured native tray coverage.

Coverage arithmetic selects candidates only. Native packing, the physical task,
demonstration, and open-loop replay remain required for every exact variant.
"""
import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for bddl in (ROOT / 'Behavior-1K/bddl3', Path('/home/minyeongk/Desktop/projects/Behavior-1K-grasp-bands-6a6194600/bddl3')):
    if bddl.exists():
        sys.path.insert(0, str(bddl))
        break
from scripts.campaign_20261110.contracts import Template, digest, require
from scripts.campaign_20261110.binding_pools import BindingPool, PoolCandidate, bind_variant, write_pool, load_pool_for_template
from scripts.campaign_20261110.bind_fresh_candidates import asset_path
from scripts.campaign_20261110.goal_groups import bind_runtime_definition, load_template_contract
from scripts.campaign_20261110.group_lane import check_reservations
from scripts.campaign_20261110.reservation_policy import active_policy

BUCKETS = ((0., .3), (.3, .45), (.45, 1.))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def count_tuples(length, maximum):
    if length == 1:
        yield from ((n,) for n in range(maximum + 1))
    else:
        for n in range(maximum + 1):
            for tail in count_tuples(length - 1, maximum - n):
                yield (n, *tail)


def solve_counts(profile, max_distractors=12, budget=.85):
    """One model per active category, repeated copies, exact native area sums."""
    require(profile.get('schema') == 'isg-native-tray-coverage-profile-v1', 'native coverage profile required')
    require(not profile.get('fixed_layout'), 'fixed puzzle layout cannot use variable counts')
    require(type(max_distractors) is int and 1 <= max_distractors <= 48, 'finite maximum count must be1..48')
    floor, target = float(profile['source_floor_area_m2']), float(profile['target']['area_m2'])
    require(math.isfinite(floor) and floor > 0 and math.isfinite(target) and target > 0, 'invalid native areas')
    models = {}
    for row in profile['distractors']:
        key = row['category'] + '/' + row['model']
        area = float(row['area_m2'])
        require(math.isfinite(area) and area > 0, 'invalid distractor area')
        if key in models:
            require(abs(area - models[key]['area_m2']) < 1e-6 and row['scale'] == models[key]['scale'], 'same model has inconsistent measured rest/area/scale')
        models[key] = row
    require(bool(models), 'no measured distractor pool')
    categories = sorted({r['category'] for r in models.values()})
    require(len(categories) <= 6, 'count solver requires a bounded category pool')
    alternatives = [[key for key in sorted(models) if models[key]['category'] == category] for category in categories]
    require(math.comb(max_distractors + len(categories), len(categories)) * math.prod(map(len, alternatives)) <= 2_000_000,
            'finite count search exceeds bounded enumeration budget')
    best = {}
    for chosen in itertools.product(*alternatives):
        for counts in count_tuples(len(chosen), max_distractors):
            n = sum(counts)
            if not 1 <= n <= max_distractors:
                continue
            occupied = target + sum(models[key]['area_m2'] * count for key, count in zip(chosen, counts))
            coverage = occupied / floor
            if coverage > budget + 1e-10:
                continue
            bucket = 0 if coverage < .3 else (1 if coverage < .45 else 2)
            centre = (.18, .375, .55)[bucket]
            selected = {key: count for key, count in zip(chosen, counts) if count}
            # Prefer the middle of a bin, then model diversity, then fewer bodies.
            rank = (abs(coverage - centre), -len(selected), n, tuple(sorted(selected.items())))
            if bucket not in best or rank < best[bucket][0]:
                best[bucket] = (rank, dict(bucket=bucket, crowdedness_range=list(BUCKETS[bucket]), counts=selected,
                    n_distractors=n, predicted_native_coverage=coverage, occupied_hull_area_m2=occupied))
    return [best[i][1] for i in sorted(best)]


def existing_pool_for_counts(template, maximum, root):
    """Keep the original maximum pool, including inactive model candidates."""
    pool = load_pool_for_template(template, root=root)
    if pool is None:
        return None
    require(not pool.fixed_layout and pool.role_limits['distractor'][0] <= 1 and maximum <= pool.role_limits['distractor'][1],
            'requested counts exceed the immutable maximum pool')
    choices = {(row.object.role, row.object.category, row.object.model): row for row in pool.candidates}
    require(len(choices) == len(pool.candidates), 'count compiler needs one default-scale choice per role/model')
    for obj in template.objects:
        if obj.role in ('target', 'distractor'):
            choice = choices[(obj.role, obj.category, obj.model)]
            require(obj.role == 'target' or choice.max_count >= maximum, 'requested count exceeds candidate copy cap')
    return pool, {key: choice.candidate_id for key, choice in choices.items()}, Path(template.scene_setup['binding_pool']['manifest'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('campaign-root', 'contracts', 'customs', 'source-bddl', 'coverage-profile', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--class-id', required=True)
    p.add_argument('--max-distractors', type=int, default=12)
    a = p.parse_args()
    require(not a.out.exists() and a.out.resolve().is_relative_to(a.campaign_root.resolve()), 'new private output required')
    pointer, reservations, hashes = active_policy(a.campaign_root)
    template = load_template_contract(a.contracts, a.class_id)
    require(template.task == 'disembedding', 'only disembedding count variants supported')
    profile = json.loads(a.coverage_profile.read_text())
    require(sha(a.source_bddl) == profile['bddl_sha256'], 'native profile BDDL bytes differ')
    match = re.search(r'\(problem\s+([^\s)]+)-(\d+)\)', a.source_bddl.read_text())
    require(match is not None, 'profile BDDL problem identity missing')
    template = bind_runtime_definition(template, profile['source_template'], runtime_metadata={
        'activity_definition': str(a.source_bddl.resolve()), 'bddl_sha256': profile['bddl_sha256'], 'problem': match[1] + '-' + match[2]})
    source_meta = json.loads((Path(template.scene_setup['source_template']) / 'template_meta.json').read_text())
    source_full = json.loads((Path(template.scene_setup['source_template']) / 'template.json').read_text())
    source_bindings = {row['name']: row for row in source_meta['bindings']}
    for obj in template.objects:
        binding = source_bindings[obj.instance]
        actual = source_full['objects_info']['init_info'][binding['scene_object_name']]['args']
        require(actual['category'] == obj.category and actual['model'] == obj.model, 'source object identity differs from ready contract')
        require(sha(asset_path(ROOT, actual)) == obj.asset_sha256, 'source USD bytes differ from ready contract')
    alias = template.scene_setup['source_alias']
    original = json.loads(a.customs.read_text())[alias]
    require(not original['generation'].get('fixed_layout') and not profile.get('fixed_layout'), 'fixed-layout LEF uses exact layout; evaluate tray variants separately')
    measured = {('target', profile['target']['category'], profile['target']['model']): profile['target']}
    measured.update({('distractor', r['category'], r['model']): r for r in profile['distractors']})
    unique = {(o.role, o.category, o.model): o for o in template.objects if o.role in {'target', 'distractor'}}
    require(set(unique) == set(measured), 'profile role/model pool differs from ready source')
    for key, obj in unique.items():
        require(max(abs(float(x) - y) for x, y in zip(measured[key]['scale'], obj.default_scale)) <= 1e-6, 'profile scale differs from ready default')
    budget_info = original['generation'].get('distractor_budget', {})
    budget = float(budget_info.get('budget_fraction', original['generation'].get('tray_budget', .85))) if isinstance(budget_info, dict) else float(budget_info)
    variants = solve_counts(profile, a.max_distractors, budget)
    require(bool(variants), 'no count candidate fits any native coverage bucket')
    fixed = tuple(o for o in template.objects if o.role == 'support')
    existing = existing_pool_for_counts(template, a.max_distractors, a.campaign_root)
    if existing:
        pool, candidate_ids, pool_path = existing
    else:
        candidate_ids = {key: 'model_' + digest({'role': obj.role, 'category': obj.category, 'model': obj.model,
            'asset_sha256': obj.asset_sha256, 'scale': obj.default_scale})[:20] for key, obj in unique.items()}
        pool = BindingPool(conceptual_template_id='disembedding_pool__' + digest({'class': template.class_id, 'split': template.split,
            'scene': template.scene, 'room': template.room, 'fixed': [asdict(o) for o in fixed], 'models': sorted(candidate_ids.values())})[:20],
            class_id=template.class_id, split=template.split, scene=template.scene, room=template.room, fixed_objects=fixed,
            candidates=tuple(PoolCandidate(candidate_ids[key], obj, 1 if obj.role == 'target' else a.max_distractors) for key, obj in sorted(unique.items())),
            role_limits={'target': (1, 1), 'distractor': (1, a.max_distractors)}, source_definition_sha256=template.definition_sha256,
            task_rooms=tuple(template.scene_setup.get('task_rooms', [template.room])), provenance={'source_contract_sha256': sha(a.contracts),
                'source_customs_sha256': sha(a.customs), 'native_coverage_profile_sha256': sha(a.coverage_profile),
                'runtime_bddl_equivalence': template.scene_setup['runtime_bddl_equivalence'], 'reservation_policy_sha256': pointer['policy_sha256']})
    check_reservations(pool.audit_template(), reservations, hashes)
    a.out.mkdir(parents=True)
    if existing:
        (a.out / 'binding_pool_reference.json').write_text(json.dumps(dict(manifest=str(pool_path), manifest_sha256=sha(pool_path), pool_sha256=pool.fingerprint), indent=2) + '\n')
    else:
        pool_path = write_pool(a.out / 'binding_pool.json', pool)
    from ispatialgym.sampling.bddl_from_iddl import RoleGrounding, render
    from ispatialgym.sampling.iddl import parse_file
    from ispatialgym.sampling.knowledge import synset_of_category
    from ispatialgym.sampling.v2.problem import GroundedProblem, iter_template_choices
    import numpy as np
    iddl = parse_file(str(ROOT / 'ispatialgym/assets/iddl' / original['iddl']))
    by_instance = {o.instance: o for o in template.objects}
    fixed_roles = {b['role']: by_instance[b['name']] for b in source_meta['bindings'] if b['name'] in by_instance and by_instance[b['name']].role == 'support'}
    target = next(o for o in unique.values() if o.role == 'target')
    templates, jobs = [], []
    for variant in variants:
        active = [(unique[('distractor', *key.split('/'))], count) for key, count in sorted(variant['counts'].items())]
        entry = copy.deepcopy(original)
        entry['roles'] = {r: entry['roles'][r] for r in iddl.roles}
        for name, obj in [('target', target)] + list(fixed_roles.items()):
            if name == 'floor':
                continue
            if entry['roles'][name].get('sampleable'):
                entry['roles'][name].update(representative=obj.model, scale={obj.model_key: list(obj.default_scale)})
        entry['roles']['distractors']['scale'] = {obj.model_key: list(obj.default_scale) for obj, _ in active}
        members = {obj.category: count for obj, count in active}
        require(len(members) == len(active), 'one exact model per distractor category is required')
        entry['generation']['groups']['distractors'] = {'size': variant['n_distractors'], 'draw': 'same_model', 'members': members}
        entry['generation']['distractor_budget'] = {'budget_fraction': budget, 'n_distractors': variant['n_distractors'],
            'source': 'native hull/rest/floor measurements; physical packing remains required', 'native_coverage_profile_sha256': sha(a.coverage_profile)}
        entry['generation'].update(templates_max=1, instances_per_template=1)
        for key in list(entry['generation']):
            if key.startswith('_'):
                entry['generation'].pop(key)
        for key in list(entry):
            if key.startswith('fallback') or key == 'superseded_by':
                entry.pop(key)
        block = copy.deepcopy(entry['scenes'][template.scene])
        block.update(rooms=list(pool.task_rooms), bindings=copy.deepcopy(source_meta['choice']['bindings']), blacklist={},
            whitelist={'target': {target.category: [target.model]}, 'distractors': {obj.category: [obj.model] for obj, _ in active}})
        for name, obj in fixed_roles.items():
            if name != 'floor' and entry['roles'][name].get('sampleable'):
                block['whitelist'][name] = {obj.category: [obj.model]}
        entry['scenes'] = {template.scene: block}; entry['constraints'] = []
        signature = '_'.join(category + '_' + str(count) for category, count in members.items())
        native_alias = iddl.variant_template.format(target=target.category, distractors=signature, support=fixed_roles['support'].category)
        did = 29000 + variant['bucket']; entry['definition_id'] = did
        groundings = {}
        for name in iddl.roles:
            if name == 'floor':
                groundings[name] = RoleGrounding(name, 'scene', same_as='floor')
            elif name == 'distractors':
                groundings[name] = RoleGrounding(name, 'group', members=[(obj.category, synset_of_category(obj.category)) for obj, count in active for _ in range(count)])
            else:
                obj = target if name == 'target' else fixed_roles[name]
                groundings[name] = RoleGrounding(name, 'sampleable' if entry['roles'][name].get('sampleable') else 'scene', category=obj.category, synset=synset_of_category(obj.category))
        rendered = render(iddl, groundings, did, native_alias, 'disembedding')
        directory = a.out / ('bucket_' + str(variant['bucket'])); directory.mkdir()
        bddl = directory / 'bddl/disembedding' / f'problem{did}.bddl'; bddl.parent.mkdir(parents=True); bddl.write_text(rendered.text)
        entry['campaign_provenance'] = {'conceptual_pool': pool.conceptual_template_id, 'native_coverage_profile_sha256': sha(a.coverage_profile),
            'source_definition_sha256': template.definition_sha256, 'count_variant': variant, 'production_released': False}
        customs = directory / 'candidate_customs.json'; customs.write_text(json.dumps({native_alias: entry}, indent=2) + '\n')
        refs = [replace(target, instance=rendered.instances['target'])]
        selection = {rendered.instances['target']: candidate_ids[('target', target.category, target.model)]}
        dist_instances = iter(rendered.instances['distractors.members'])
        for obj, count in active:
            for _ in range(count):
                instance = next(dist_instances); refs.append(replace(obj, instance=instance)); selection[instance] = candidate_ids[('distractor', obj.category, obj.model)]
        for name, obj in fixed_roles.items():
            require(rendered.instances[name] == obj.instance, 'count variant changed a fixed support binding')
            refs.append(obj)
        setup = {k: v for k, v in template.scene_setup.items() if k not in {'source_template', 'source_template_sha256', 'source_bddl_sha256',
            'source_alias', 'runtime_template_id', 'inst_to_name', 'runtime_bddl_equivalence', 'fresh_metadata_sha256', 'actual_init_args'}}
        setup.update(runtime_bddl_sha256=sha(bddl), customs_alias=native_alias, coverage_candidate=variant, native_coverage_profile_sha256=sha(a.coverage_profile),
            fresh_geometry='pending', fresh_demo='pending', fresh_replay='pending', release_status='candidate_only')
        physical = Template(template_id=pool.conceptual_template_id+'__count_'+digest(variant['counts'])[:16], task=template.task,
            class_id=template.class_id, split=template.split, scene=template.scene, room=template.room, objects=tuple(refs),
            definition_sha256=digest({'entry': entry, 'bddl_sha256': sha(bddl)}), scene_setup=setup)
        physical = bind_variant(physical, pool_path, selection)
        check_reservations(physical, reservations, hashes)
        templates.append({**asdict(physical), 'fingerprint': physical.fingerprint})
        gp = GroundedProblem.load(native_alias, str(customs), str(directory / 'bddl'))
        require(gp.rooms(template.scene) == list(pool.task_rooms), 'count variant native room mismatch')
        choices = list(iter_template_choices(gp, template.scene, np.random.default_rng(20261110), max_templates=1))
        require(len(choices) == 1, 'native count variant enumerates no template')
        actual_members = [(x.category, x.model) for x in choices[0].groups['distractors']]
        require(actual_members == [(obj.category, obj.model) for obj, count in active for _ in range(count)], 'native count/model enumeration differs')
        jobs.append(dict(candidate_id=physical.template_id, class_id=template.class_id, task=template.task, alias=native_alias,
            scene=template.scene, room=template.room, split=template.split, template_fingerprint=physical.fingerprint,
            count_variant=variant, production_released=False, source_manifest=str((a.out / 'validation_jobs.json').resolve()),
            sample_argv=['python', '-m', 'ispatialgym.sampling.sample_isg', '--variant', native_alias, '--customs', str(customs.resolve()),
                '--bddl-root', str((directory / 'bddl').resolve()), '--scene', template.scene, '--templates', '1', '--instances-per-template', '1',
                '--max-attempts', '3', '--seed', '20261110', '--no-write-back', '--out', '${CAMPAIGN_OUT}/validation/' + physical.template_id]))
    for filename, data in [('contract_candidates.json', {'templates': templates}), ('validation_jobs.json', {'jobs': jobs}),
            ('count_solver_report.json', {'variants': variants, 'missing_buckets': sorted(set(range(3)) - {v['bucket'] for v in variants}),
                'coverage_profile_sha256': sha(a.coverage_profile), 'scope': 'Exact native area prediction; packing and task validation pending.'})]:
        (a.out / filename).write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'variants': len(jobs), 'pool_sha256': pool.fingerprint, 'manifest_sha256': sha(a.out / 'validation_jobs.json')}))


if __name__ == '__main__':
    main()
