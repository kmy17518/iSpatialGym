#!/usr/bin/env python3
"""Rebind the camera pot-plant holdout after audited monotonic reservation adoption."""
import argparse
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path

from scripts.campaign_20261110.bind_fresh_candidates import load_contract
from scripts.campaign_20261110.contracts import ObjectRef, digest, file_digest, require, write_json_atomic
from scripts.campaign_20261110.group_lane import check_reservations
from scripts.campaign_20261110.reservation_policy import active_policy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'source', 'defaults', 'visual-review', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args(); root = a.root.resolve()
    require(not a.out.exists() and a.out.resolve().is_relative_to(root), 'new owned output required')
    pointer, policy, hashes = active_policy(root)
    key = 'pot_plant/tbsjnw'; old = 'pot_plant/udqjui'; cls = 'camera_relocalization-obstructed-pot_plant_1'
    require(key in policy['model_holdout_models'] and old in policy['model_holdout_models'], 'monotonic replacement not adopted')
    defaults, review = [json.loads(path.read_text()) for path in (a.defaults,a.visual_review)]
    row = next(x for x in defaults['objects'] if x['category']+'/'+x['model'] == key)
    require(defaults['status'] == 'native_defaults_and_geometry_recorded' and row['ready'], 'native ready default missing')
    require(review['default_geometry_sha256'] == file_digest(a.defaults) and review['objects'][key]['status'] == 'passed', 'visual review missing')
    require(row['asset_sha256'] == hashes['ready_assets'][key]['asset_sha256'] == file_digest(row['asset_path']), 'asset identity differs')
    original = next(t for t in json.loads((a.source/'contract_candidates.json').read_text())['templates'] if t['class_id'] == cls)
    template = load_contract(original)
    job = copy.deepcopy(next(j for j in json.loads((a.source/'validation_jobs.json').read_text())['jobs'] if j['template_fingerprint'] == template.fingerprint))
    alias = job['alias']; entry = copy.deepcopy(json.loads((a.source/'candidate_customs.json').read_text())[alias])
    require(entry['roles']['obstacles']['representative'] == 'udqjui', 'unexpected source model')
    entry['roles']['obstacles'].update(representative='tbsjnw',scale={key:row['default_scale']})
    entry['scenes'][template.scene]['whitelist']['obstacles'] = {'pot_plant':['tbsjnw']}
    proof = dict(active_policy_sha256=pointer['policy_sha256'], source_candidate_fingerprint=template.fingerprint,
        source_jobs_sha256=file_digest(a.source/'validation_jobs.json'), native_defaults_sha256=file_digest(a.defaults),
        native_visual_review_sha256=file_digest(a.visual_review), replaced_model_still_reserved=old, production_released=False)
    entry['campaign_provenance'] = proof
    refs = []
    for obj in template.objects:
        if obj.model_key == old:
            refs.append(ObjectRef(instance=obj.instance,role=obj.role,category='pot_plant',model='tbsjnw',asset_sha256=row['asset_sha256'],
                default_scale=tuple(row['default_scale']),ready=True,readiness_source=row['readiness_source']))
        else:
            refs.append(obj)
    require(sum(obj.model_key == key for obj in refs) == 1, 'expected exactly one replacement obstacle')
    a.out.mkdir(parents=True)
    bddl = a.out/'bddl/camera_relocalization'/f'problem{entry["definition_id"]}.bddl'; bddl.parent.mkdir(parents=True)
    source_bddl = a.source/'bddl/camera_relocalization'/bddl.name
    require(file_digest(source_bddl) == template.scene_setup['runtime_bddl_sha256'], 'source BDDL differs')
    bddl.write_bytes(source_bddl.read_bytes())
    customs = a.out/'candidate_customs.json';write_json_atomic(customs,{alias:entry})
    setup = {k:v for k,v in template.scene_setup.items() if k not in {'default_geometry_sha256','model_visual_review_sha256','reservation_sha256'}}
    setup.update(proof,default_geometry_sha256=file_digest(a.defaults),model_visual_review_sha256=file_digest(a.visual_review),fresh_geometry='pending')
    bound = replace(template,template_id='held_model__camera_pot_plant_tbsjnw_r6',objects=tuple(refs),
        definition_sha256=digest(dict(entry=entry,bddl_sha256=file_digest(bddl))),scene_setup=setup)
    check_reservations(bound,policy,hashes)
    from ispatialgym.sampling.v2.problem import GroundedProblem,iter_template_choices
    import numpy as np
    native = GroundedProblem.load(alias,str(customs),str(a.out/'bddl'))
    choices = list(iter_template_choices(native,template.scene,np.random.default_rng(20261110),max_templates=1))
    require(len(choices)==1 and [(x.category,x.model) for x in choices[0].groups['obstacles']] == [('pot_plant','tbsjnw')], 'native obstacle enumeration differs')
    job.update(candidate_id=bound.template_id,template_fingerprint=bound.fingerprint,source_manifest=str(a.out/'validation_jobs.json'),
        sample_argv=[x.replace('${MATERIALIZED}',str(a.out)) for x in job['sample_argv']],replaced_assignment_proof=proof)
    write_json_atomic(a.out/'validation_jobs.json',{'jobs':[job]})
    write_json_atomic(a.out/'contract_candidates.json',{'templates':[{**asdict(bound),'fingerprint':bound.fingerprint}]})
    print(json.dumps(dict(candidates=1,manifest_sha256=file_digest(a.out/'validation_jobs.json'),fingerprint=bound.fingerprint)))


if __name__ == '__main__':
    main()
