#!/usr/bin/env python3
"""Prepare count canaries from native CPU choices; physical validation is pending.

The metadata seed is explicitly not a sampled scene. Native multiply_isg builds
the scene from the exact TemplateChoice, then writes fresh physical metadata.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys

from scripts.campaign_20261110.bind_fresh_candidates import load_contract
from scripts.campaign_20261110.binding_pools import load_pool_for_template
from scripts.campaign_20261110.contracts import Perturbations, ScaleState, ValidationKey, file_digest, require, write_json_atomic
from scripts.campaign_20261110.group_lane import check_reservations
from scripts.campaign_20261110.reservation_policy import active_policy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'snapshot', 'contracts', 'jobs', 'accepted-anchor', 'physics', 'data-identity', 'v2-root', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    root, snapshot, out = a.root.resolve(), a.snapshot.resolve(), a.out.resolve()
    require(out.is_relative_to(root / 'groups') and not out.exists(), 'new owned groups directory required')
    require(snapshot.is_relative_to(root / 'code'), 'runtime outside campaign')
    policy_record, policy, hashes = active_policy(root)
    anchor_doc = json.loads(a.accepted_anchor.read_text())
    anchor = anchor_doc['template']
    anchor_pool = anchor['scene_setup']['binding_pool']
    fixed = {o['instance']: o for o in anchor['objects'] if o['role'] in ('support', 'target')}
    templates = [load_contract(row) for row in json.loads(a.contracts.read_text())['templates']]
    jobs = json.loads(a.jobs.read_text())['jobs']
    require(len(templates) == len(jobs) == 3, 'expected exactly three count variants')
    require(sorted(sum(o.role == 'distractor' for o in t.objects) for t in templates) == [5, 8, 12], 'unexpected count set')
    from ispatialgym.sampling.v2.problem import GroundedProblem, iter_template_choices
    from ispatialgym.sampling.multiply_isg import problem_for_template, choice_for_template
    import numpy as np
    out.mkdir(parents=True)
    prepared = []
    for template in templates:
        check_reservations(template, policy, hashes)
        pool = load_pool_for_template(template, root=root)
        require(pool is not None and pool.fingerprint == anchor_pool['pool_sha256'], 'conceptual pool differs from accepted anchor')
        require(template.scene_setup['binding_pool']['manifest_sha256'] == anchor_pool['manifest_sha256'], 'maximum pool file changed')
        require({o.instance: json.loads(json.dumps(asdict(o))) for o in template.objects if o.role in ('support', 'target')} == fixed,
                'fixed target or original source/destination/support differs from accepted anchor')
        matches = [j for j in jobs if j['template_fingerprint'] == template.fingerprint]
        require(len(matches) == 1, 'ambiguous count command')
        job = matches[0]
        argv = job['sample_argv']
        customs = Path(argv[argv.index('--customs') + 1]).resolve()
        bddl_root = Path(argv[argv.index('--bddl-root') + 1]).resolve()
        require(customs.is_relative_to(root) and bddl_root.is_relative_to(root), 'private inputs outside campaign')
        entry = json.loads(customs.read_text())[job['alias']]
        bddl = bddl_root / 'disembedding' / ('problem' + str(entry['definition_id']) + '.bddl')
        require(file_digest(bddl) == template.scene_setup['runtime_bddl_sha256'], 'private BDDL differs')
        gp = GroundedProblem.load(job['alias'], str(customs), str(bddl_root))
        choices = list(iter_template_choices(gp, template.scene, np.random.default_rng(20261110), max_templates=1))
        require(len(choices) == 1, 'native model choice unavailable')
        choice = choices[0]
        require(Counter((m.category, m.model) for m in choice.groups['distractors']) ==
                Counter((o.category, o.model) for o in template.objects if o.role == 'distractor'), 'native distractor multiplicity differs')
        count = sum(o.role == 'distractor' for o in template.objects)
        directory = out / ('count_' + str(count))
        seed = directory / 'cpu_choice_seed' / choice.template_id
        seed.mkdir(parents=True)
        metadata = dict(schema='isg-campaign-cpu-choice-seed-v1', variant=job['alias'], scene=template.scene,
                        template_id=choice.template_id, task_rooms=list(choice.rooms), choice=choice.to_json(), customs=str(customs),
                        problem='disembedding-' + str(entry['definition_id']), activity_definition=str(bddl), bddl_sha256=file_digest(bddl),
                        n_written=0, physical_geometry_validated=False,
                        provenance=dict(native_api='iter_template_choices + choice_for_template', candidate_fingerprint=template.fingerprint,
                                        accepted_pool_anchor=str(a.accepted_anchor), accepted_pool_anchor_sha256=file_digest(a.accepted_anchor)))
        write_json_atomic(seed / 'template_meta.json', metadata)
        # Exercise exactly the two read-only native loader functions used by multiply_isg.
        loaded = problem_for_template(metadata, str(customs), str(bddl_root))
        require(choice_for_template(loaded, metadata).to_json() == choice.to_json(), 'native seed roundtrip differs')
        perturbations = Perturbations({o.instance: ScaleState(o.default_scale, o.default_scale, 'fixed') for o in template.objects})
        key = ValidationKey(template.fingerprint, file_digest(snapshot / '.snapshot_commit'), file_digest(a.physics),
                            file_digest(a.data_identity), perturbations.scale_fingerprint)
        config = dict(campaign_id='campaign_20261110', split=template.split, purpose='validation', template=asdict(template),
                      perturbations=asdict(perturbations), validation_key=asdict(key))
        write_json_atomic(directory / 'campaign_config.json', config)
        write_json_atomic(directory / 'source.json', dict(template_dir=str(seed), anchor_is_final_member=False,
                          source_kind='native_cpu_enumerated_choice_seed', physical_geometry_validated=False,
                          candidate_jobs=str(a.jobs), candidate_jobs_sha256=file_digest(a.jobs), source_candidate=job,
                          active_policy_sha256=policy_record['policy_sha256'], runtime_bddl=str(bddl), runtime_bddl_sha256=file_digest(bddl)))
        write_json_atomic(directory / 'empty_eval_index.json', dict(schema='isg-instance-index-v1', n=0, instances=[]))
        command = [sys.executable, str(snapshot / 'scripts/campaign_20261110/infra_submit_validation.py'),
                   '--root', str(root), '--group', str(directory), '--snapshot', str(snapshot), '--key', template.class_id,
                   '--tag', 'disembedding02_count_canary_v1', '--v2-root', str(a.v2_root), '--index', '${FRESH_GLOBAL_INDEX}',
                   '--template-customs', '--bddl-root', str(bddl_root), '--customs-file', str(customs), '--kcollect', '1']
        prepared.append(dict(candidate_id=template.template_id, class_id=template.class_id, count=count, directory=str(directory),
                             status='prepared_requires_fresh_index', command=command, template_fingerprint=template.fingerprint,
                             native_choice_roundtrip='passed', config_sha256=file_digest(directory / 'campaign_config.json'),
                             native_seed_sha256=file_digest(seed / 'template_meta.json'), customs_sha256=file_digest(customs),
                             source_job=job, production_released=False, physical_geometry_validated=False))
    path = out / 'prepared_jobs.json'
    write_json_atomic(path, dict(schema='isg-native-count-canaries-v1', active_policy_sha256=policy_record['policy_sha256'],
                      source_contracts_sha256=file_digest(a.contracts), source_jobs_sha256=file_digest(a.jobs),
                      pool_sha256=anchor_pool['pool_sha256'], jobs=prepared))
    print(json.dumps(dict(prepared=len(prepared), manifest=str(path), sha256=file_digest(path))))


if __name__ == '__main__':
    main()
