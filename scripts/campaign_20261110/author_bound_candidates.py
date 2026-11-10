#!/usr/bin/env python3
"""Prepare exact native-bound validation configs without reserving or launching jobs."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from scripts.campaign_20261110.bind_fresh_candidates import load_contract
from scripts.campaign_20261110.contracts import Perturbations, ScaleState, ValidationKey, file_digest, require, write_json_atomic
from scripts.campaign_20261110.goal_groups import bind_runtime_definition
from scripts.campaign_20261110.group_lane import check_reservations
from scripts.campaign_20261110.reservation_policy import active_policy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'snapshot', 'contracts', 'jobs', 'physics', 'data-identity', 'v2-root', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    root, snapshot = a.root.resolve(), a.snapshot.resolve()
    require(not a.out.exists() and a.out.resolve().is_relative_to(root / 'groups'), 'new owned group batch required')
    require(snapshot.is_relative_to(root / 'code'), 'snapshot outside campaign')
    active = active_policy(root); require(active is not None, 'committed policy missing')
    policy_record, policy, asset_hashes = active
    policy_hash = policy_record['policy_sha256']
    contracts = json.loads(a.contracts.read_text())
    jobs = json.loads(a.jobs.read_text())['jobs']
    templates = sorted((load_contract(row) for row in contracts['templates']), key=lambda t: (t.split != 'eval_model', t.class_id, t.template_id))
    require(not contracts.get('failures'), 'binding report contains failed proofs')
    a.out.mkdir(parents=True)
    results = []
    for template in templates:
        result = dict(candidate_id=template.template_id, class_id=template.class_id, split=template.split,
            source_fingerprint=template.fingerprint, status='blocked')
        results.append(result)
        try:
            check_reservations(template, policy, asset_hashes)
            matches = [job for job in jobs if job['template_fingerprint'] == template.scene_setup['candidate_fingerprint']]
            require(len(matches) == 1, 'ambiguous candidate command source')
            job = matches[0]; argv = job['sample_argv']
            bddl_root = Path(argv[argv.index('--bddl-root') + 1]).resolve()
            customs = Path(argv[argv.index('--customs') + 1]).resolve()
            require(bddl_root.is_relative_to(root) and bddl_root.is_dir(), 'private BDDL root not owned')
            require(customs.is_relative_to(root) and customs.is_file(), 'private customs not owned')
            source = Path(template.scene_setup['source_template']).resolve()
            meta = json.loads((source / 'template_meta.json').read_text())
            require(Path(meta['customs']).resolve() == customs and meta['variant'] in json.loads(customs.read_text()), 'native private customs differs')
            activity, number = meta['problem'].rsplit('-', 1)
            bddl = bddl_root / activity / ('problem' + number + '.bddl')
            require(file_digest(bddl) == template.scene_setup['runtime_bddl_sha256'], 'private runtime BDDL differs')
            meta.update(activity_definition=str(bddl), bddl_sha256=file_digest(bddl))
            bound = bind_runtime_definition(template, source, runtime_metadata=meta)
            check_reservations(bound, policy, asset_hashes)
            perturbations = Perturbations({obj.instance: ScaleState(obj.default_scale, obj.default_scale, 'fixed') for obj in bound.objects})
            validation = ValidationKey(bound.fingerprint, file_digest(snapshot / '.snapshot_commit'), file_digest(a.physics), file_digest(a.data_identity), perturbations.scale_fingerprint)
            directory = a.out / bound.template_id
            require(directory.resolve().is_relative_to(a.out.resolve()), 'unsafe template ID')
            directory.mkdir()
            config = dict(campaign_id='campaign_20261110', split=bound.split, purpose='validation', template=asdict(bound),
                perturbations=asdict(perturbations), validation_key=asdict(validation))
            write_json_atomic(directory / 'campaign_config.json', config)
            write_json_atomic(directory / 'source.json', dict(template_dir=str(source), anchor_is_final_member=False,
                bound_contract_manifest=str(a.contracts), bound_contract_manifest_sha256=file_digest(a.contracts),
                candidate_jobs=str(a.jobs), candidate_jobs_sha256=file_digest(a.jobs), source_candidate=job,
                active_policy_sha256=policy_hash, runtime_bddl=str(bddl), runtime_bddl_sha256=file_digest(bddl)))
            write_json_atomic(directory / 'empty_eval_index.json', dict(schema='isg-instance-index-v1', n=0, instances=[]))
            command = [sys.executable, str(snapshot / 'scripts/campaign_20261110/infra_submit_validation.py'),
                '--root', str(root), '--group', str(directory), '--snapshot', str(snapshot), '--key', bound.class_id,
                '--tag', 'floating_annotated_r2', '--v2-root', str(a.v2_root), '--index', '${FRESH_GLOBAL_INDEX}',
                '--template-customs', '--bddl-root', str(bddl_root), '--customs-file', str(customs)]
            result.update(status='prepared_requires_fresh_index', directory=str(directory), command=command,
                fingerprint=bound.fingerprint, config_sha256=file_digest(directory / 'campaign_config.json'),
                validation_key=asdict(validation), runtime_bddl_sha256=file_digest(bddl), production_released=False)
        except Exception as exc:
            result['error'] = str(exc)
    path = a.out / 'prepared_jobs.json'
    write_json_atomic(path, dict(contracts_sha256=file_digest(a.contracts), jobs_sha256=file_digest(a.jobs),
        active_policy_sha256=policy_hash, jobs=results))
    print(json.dumps(dict(prepared=sum(row['status'] == 'prepared_requires_fresh_index' for row in results),
        blocked=[row for row in results if row['status'] == 'blocked'], manifest=str(path), sha256=file_digest(path))))


if __name__ == '__main__':
    main()
