"""Clone unseen pair candidates with directly reviewed, model-specific fronts."""
import argparse
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil

from scripts.campaign_20261110.bind_fresh_candidates import load_contract
from scripts.campaign_20261110.contracts import digest, file_digest, require, write_json_atomic
from scripts.campaign_20261110.group_lane import check_reservations
from scripts.campaign_20261110.reservation_policy import active_policy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'source', 'review', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--class-id', action='append', required=True)
    a = p.parse_args()
    require(not a.out.exists() and a.out.resolve().is_relative_to(a.root.resolve()), 'new owned output required')
    _, policy, hashes = active_policy(a.root)
    review = json.loads(a.review.read_text())
    require(review['status'] == 'reviewed_selected_semantic_fronts', 'semantic front review missing')
    source = Path(review['native_geometry_report']['path'])
    require(file_digest(source) == review['native_geometry_report']['sha256'], 'native render evidence changed')
    native = {r['category'] + '/' + r['model']: r for r in json.loads(source.read_text())['objects']}
    for key, row in review['objects'].items():
        measured_native = native
        if row.get('native_geometry_report'):
            record = row['native_geometry_report']; extra = Path(record['path'])
            require(extra.resolve().is_relative_to(a.root.resolve()) and file_digest(extra) == record['sha256'], 'object render proof changed or outside campaign')
            extra_doc = json.loads(extra.read_text())
            require(extra_doc['status'] == 'native_defaults_and_geometry_recorded', 'object render probe incomplete')
            measured_native = {r['category'] + '/' + r['model']: r for r in extra_doc['objects']}
        require(row['front'] in ('+x', '-x', '+y', '-y') and row['status'] == 'passed', 'invalid reviewed signed front')
        require(row['asset_sha256'] == measured_native[key]['asset_sha256'], 'reviewed asset differs from native render')
        measured = {(im['path'], im['sha256']) for im in measured_native[key]['images']}
        require(row['images'] and row['findings'], 'semantic review evidence absent')
        for image in row['images']:
            require((image['path'], image['sha256']) in measured and file_digest(Path(image['path'])) == image['sha256'], 'review image changed')
    contracts = {r['class_id']: load_contract(r) for r in json.loads((a.source / 'contract_candidates.json').read_text())['templates']}
    original = json.loads((a.source / 'candidate_customs.json').read_text())
    customs, jobs, templates = {}, [], []
    for old in json.loads((a.source / 'validation_jobs.json').read_text())['jobs']:
        if old['class_id'] not in a.class_id:
            continue
        template = contracts[old['class_id']]
        check_reservations(template, policy, hashes)
        entry = copy.deepcopy(original[old['alias']]); spec = entry['generation'].get('pair_goal_align')
        require(isinstance(spec, dict), 'source lacks pair alignment semantics')
        keys = {o.model_key for o in template.objects if o.role in ('target', 'partner')}
        require(keys <= review['objects'].keys(), 'pair lacks direct front review')
        spec['front'] = {key: review['objects'][key]['front'] for key in sorted(keys)}
        proof = dict(path=str(a.review), sha256=file_digest(a.review), source_customs_sha256=file_digest(a.source / 'candidate_customs.json'))
        entry.setdefault('campaign_provenance', {})['semantic_front_review'] = proof
        updated = replace(template, template_id=template.template_id + '__reviewed_fronts_v1',
            definition_sha256=digest(dict(source_definition=template.definition_sha256, entry=entry, review=proof)),
            scene_setup={**template.scene_setup, 'semantic_front_review': proof})
        customs[old['alias']] = entry
        job = copy.deepcopy(old);job.update(candidate_id=updated.template_id, template_fingerprint=updated.fingerprint)
        job['sample_argv'] = [x.replace(str(a.source), '${MATERIALIZED}') for x in job['sample_argv']]
        jobs.append(job);templates.append({**asdict(updated), 'fingerprint': updated.fingerprint})
    require(len(jobs) == len(set(a.class_id)), 'requested class missing or duplicated')
    a.out.mkdir(parents=True);shutil.copytree(a.source / 'bddl', a.out / 'bddl')
    write_json_atomic(a.out / 'candidate_customs.json', customs)
    from ispatialgym.sampling.v2.problem import GroundedProblem, iter_template_choices
    import numpy as np
    for job in jobs:
        gp = GroundedProblem.load(job['alias'], str(a.out / 'candidate_customs.json'), str(a.out / 'bddl'))
        choices = list(iter_template_choices(gp, job['scene'], np.random.default_rng(20261110), max_templates=1))
        require(len(choices) == 1, 'native model uniqueness/lookalike check yields no pair')
        for role in ('target', 'partner'):
            model = choices[0].models[role]
            require(gp.gen('pair_goal_align')['front'][model.category + '/' + model.model] in ('+x', '-x', '+y', '-y'), 'native front absent')
    write_json_atomic(a.out / 'validation_jobs.json', dict(jobs=jobs, production_released=False))
    write_json_atomic(a.out / 'contract_candidates.json', dict(templates=templates))
    print(json.dumps(dict(candidates=len(jobs), manifest=str(a.out / 'validation_jobs.json'), sha256=file_digest(a.out / 'validation_jobs.json'))))


if __name__ == '__main__':
    main()
