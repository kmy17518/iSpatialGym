"""Create an isolated UNDER pair using ready, measured reserved stationery."""
import argparse
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path

from scripts.campaign_20261110.bind_fresh_candidates import load_contract
from scripts.campaign_20261110.contracts import Template, digest, file_digest, require, write_json_atomic
from scripts.campaign_20261110.group_lane import check_reservations
from scripts.campaign_20261110.reservation_policy import active_policy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'source', 'out', 'snapshot'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    require(not a.out.exists() and a.out.resolve().is_relative_to(a.root.resolve()), 'new owned output required')
    _, policy, hashes = active_policy(a.root)
    templates = [load_contract(x) for x in json.loads((a.source / 'contract_candidates.json').read_text())['templates']]
    source = next(t for t in templates if t.task == 'inter_object_relation_satisfaction-under_small_scale')
    pair = next(t for t in templates if t.task == 'inter_object_relation_satisfaction-ontop_small_scale')
    require([(o.category, o.model) for o in pair.objects if o.role == 'target'] == [('rubber_eraser', 'cungod')], 'unexpected measured target')
    require([(o.category, o.model) for o in pair.objects if o.role == 'partner'] == [('notepad', 'awcyha')], 'unexpected measured partner')
    check_reservations(pair, policy, hashes)
    jobs = json.loads((a.source / 'validation_jobs.json').read_text())['jobs']
    old = next(j for j in jobs if j['class_id'] == source.class_id)
    donor = next(j for j in jobs if j['class_id'] == pair.class_id)
    customs = json.loads((a.source / 'candidate_customs.json').read_text())
    entry = copy.deepcopy(customs[old['alias']]); native_pair = customs[donor['alias']]
    for role in ('target', 'partner'):
        entry['roles'][role] = copy.deepcopy(native_pair['roles'][role])
    for scene, spec in entry['scenes'].items():
        for role in ('target', 'partner'):
            spec['whitelist'][role] = copy.deepcopy(native_pair['scenes'][scene]['whitelist'][role])
    from ispatialgym.sampling.iddl import parse_file
    from ispatialgym.sampling.bddl_from_iddl import RoleGrounding, render
    from ispatialgym.sampling.knowledge import synset_of_category
    from ispatialgym.sampling.v2.problem import GroundedProblem, iter_template_choices
    import numpy as np
    iddl = parse_file(str(a.snapshot / 'ispatialgym/assets/iddl' / entry['iddl']))
    alias = iddl.variant_template.format(target='rubber_eraser', partner='notepad', support=entry['roles']['support']['category'])
    entry['definition_id'] = 29910
    refs_by_role = {o.role: o for o in pair.objects if o.role in ('target', 'partner')}
    roles = {}
    for role in iddl.roles:
        if role == 'floor':
            roles[role] = RoleGrounding(role, 'scene', same_as='floor')
        else:
            category = entry['roles'][role]['category']
            roles[role] = RoleGrounding(role, 'sampleable' if role in refs_by_role else 'scene', category=category, synset=synset_of_category(category))
    bddl = render(iddl, roles, 29910, alias, 'inter_object_relation_satisfaction')
    a.out.mkdir(parents=True)
    definition = a.out / 'bddl/inter_object_relation_satisfaction/problem29910.bddl';definition.parent.mkdir(parents=True);definition.write_text(bddl.text)
    objects = []
    for obj in pair.objects:
        role = obj.role if obj.role in refs_by_role else 'floor' if obj.semantic_kind == 'floor' else 'support'
        objects.append(replace(obj, instance=bddl.instances[role]))
    proof = dict(source_under_definition=source.definition_sha256, measured_pair_fingerprint=pair.fingerprint,
                 source_customs=str(a.source / 'candidate_customs.json'), source_customs_sha256=file_digest(a.source / 'candidate_customs.json'),
                 semantics='Native UNDER small: the partner notepad ends under the manipulated eraser; original task classes unchanged.')
    entry['campaign_provenance'] = {**entry.get('campaign_provenance', {}), 'isolated_under_pair': proof}
    cls = source.task + '-rubber_eraser-notepad'
    setup = {**pair.scene_setup, 'runtime_bddl_sha256': file_digest(definition), 'customs_alias': alias, 'isolated_under_pair': proof}
    # Fixed scene role names remain identical while sampleable BDDL names change.
    setup['inst_to_name'] = {o.instance: pair.scene_setup['inst_to_name'][old_obj.instance]
                            for o, old_obj in zip(objects, pair.objects) if old_obj.instance in pair.scene_setup['inst_to_name']}
    template = Template('held_category__' + cls, source.task, cls, 'eval_category', pair.scene, pair.room,
                        tuple(objects), digest(dict(entry=entry, bddl_sha256=file_digest(definition))), setup)
    check_reservations(template, policy, hashes)
    path = a.out / 'candidate_customs.json';write_json_atomic(path, {alias: entry})
    gp = GroundedProblem.load(alias, str(path), str(a.out / 'bddl'))
    require(len(list(iter_template_choices(gp, pair.scene, np.random.default_rng(20261110), max_templates=1))) == 1, 'native pair enumeration failed')
    job = {**copy.deepcopy(donor), 'candidate_id': template.template_id, 'class_id': cls, 'task': source.task, 'alias': alias,
           'template_fingerprint': template.fingerprint, 'source_manifest': str(a.out / 'validation_jobs.json')}
    job['sample_argv'] = ['python', '-m', 'ispatialgym.sampling.sample_isg', '--variant', alias, '--customs', str(path),
        '--bddl-root', str(a.out / 'bddl'), '--scene', pair.scene, '--templates', '1', '--instances-per-template', '1',
        '--max-attempts', '3', '--seed', '20261110', '--no-write-back', '--out', '${CAMPAIGN_OUT}/validation/' + cls]
    write_json_atomic(a.out / 'contract_candidates.json', {'templates': [{**asdict(template), 'fingerprint': template.fingerprint}]})
    write_json_atomic(a.out / 'validation_jobs.json', {'jobs': [job], 'production_released': False})
    print(json.dumps(dict(manifest=str(a.out / 'validation_jobs.json'), sha256=file_digest(a.out / 'validation_jobs.json'))))


if __name__ == '__main__':
    main()
