#!/usr/bin/env python3
"""Clone only floating consumers whose audited native alias changed."""
import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for path in (ROOT / 'Behavior-1K/bddl3', Path('/home/minyeongk/Desktop/projects/Behavior-1K-grasp-bands-6a6194600/bddl3')):
    if path.exists():
        sys.path.insert(0, str(path))
from scripts.campaign_20261110.bind_fresh_candidates import load_contract
from scripts.campaign_20261110.contracts import digest, require
from scripts.campaign_20261110.group_lane import check_reservations
from scripts.campaign_20261110.reservation_policy import active_policy
from ispatialgym.sampling.v2.problem import GroundedProblem


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_alias_only(old, new, alias):
    require({k: v for k, v in old.items() if k != 'legacy_keys'} ==
            {k: v for k, v in new.items() if k != 'legacy_keys'}, 'floating physical payload changed')
    require(set(old.get('legacy_keys', [])) <= set(new.get('legacy_keys', [])), 'existing routing aliases removed')
    require(alias in new.get('legacy_keys', []), 'new native alias absent')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('campaign-root', 'jobs', 'alias-audit', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    require(not a.output.exists() and a.output.resolve().is_relative_to(a.campaign_root.resolve()), 'new private output required')
    pointer, reservations, hashes = active_policy(a.campaign_root)
    audit = json.loads(a.alias_audit.read_text())
    require(audit['native_task_reader_errors'] == 0 and audit['physical_grasps_rests_geometry_unchanged'], 'native alias audit failed')
    routes = {r['id']: r for r in audit['records']}
    previous_release = Path(audit['previous_release'])
    current_release = a.alias_audit.parent
    originals = json.loads(a.jobs.read_text())['jobs']
    jobs, templates, unchanged = [], [], []
    a.output.mkdir(parents=True)
    for index, source_job in enumerate(originals):
        dep = source_job['dependency_library']; route = routes[dep['id']]
        require(dep['library_sha256'] == route['source_library_sha256'] and dep['alias'] == route['old_alias'], 'source consumer alias/library audit differs')
        oldlib = previous_release / 'glib/models' / dep['id'] / 'library.json'
        newlib = current_release / 'glib/models' / dep['id'] / 'library.json'
        require(sha(oldlib) == route['source_library_sha256'] and sha(newlib) == route['library_sha256'], 'audited library bytes changed')
        if route['old_alias'] == route['alias']:
            require(sha(oldlib) == sha(newlib), 'unchanged alias unexpectedly changed library')
            unchanged.append(dict(candidate_id=source_job['candidate_id'], fingerprint=source_job['template_fingerprint'], reason='native alias and exact library unchanged'))
            continue
        verify_alias_only(json.loads(oldlib.read_text()), json.loads(newlib.read_text()), route['alias'])
        source_contracts = Path(source_job['source_manifest']).parent / 'contract_candidates.json'
        raw = next(r for r in json.loads(source_contracts.read_text())['templates'] if r['fingerprint'] == source_job['template_fingerprint'])
        template = load_contract(raw)
        require(template.template_id == source_job['candidate_id'], 'source candidate identity differs')
        targets = [obj for obj in template.objects if obj.role == 'target']
        require(len(targets) == 1, 'floating candidate requires exactly one target')
        target = targets[0]; key = target.category + '/' + target.model
        require(target.asset_sha256 == route['asset_sha256'] and list(target.default_scale) == route['scale'], 'alias belongs to another asset/scale')
        require(route['alias'] == target.category + '__' + target.model + '__f', 'unexpected native floating alias')
        argv = list(source_job['sample_argv'])
        source_customs = Path(argv[argv.index('--customs') + 1]); bddl_root = Path(argv[argv.index('--bddl-root') + 1])
        entry = copy.deepcopy(json.loads(source_customs.read_text())[source_job['alias']])
        require(entry['roles']['target']['grasp_keys'][key] == route['old_alias'], 'source target routing differs from audit')
        entry['roles']['target']['grasp_keys'][key] = route['alias']
        directory = a.output / f'{index:02d}'; directory.mkdir()
        customs = directory / 'candidate_customs.json'; customs.write_text(json.dumps({source_job['alias']: entry}, indent=2) + '\n')
        bddl_sha = template.scene_setup['runtime_bddl_sha256']
        matches = [path for path in bddl_root.rglob('*.bddl') if sha(path) == bddl_sha]
        require(len(matches) == 1, 'exact source BDDL absent or ambiguous')
        bddl = directory / 'bddl' / matches[0].relative_to(bddl_root)
        bddl.parent.mkdir(parents=True); bddl.write_bytes(matches[0].read_bytes())
        proof = dict(source_candidate_fingerprint=template.fingerprint, source_jobs_sha256=sha(a.jobs), source_customs_sha256=sha(source_customs),
            alias_audit_sha256=sha(a.alias_audit), old_alias=route['old_alias'], native_alias=route['alias'], source_library_sha256=sha(oldlib),
            library_sha256=sha(newlib), physical_payload_unchanged=True, runtime_bddl_unchanged=True)
        setup = {k: v for k, v in template.scene_setup.items() if k not in {'source_template', 'source_template_sha256', 'source_bddl_sha256',
            'source_alias', 'runtime_template_id', 'inst_to_name', 'runtime_bddl_equivalence', 'fresh_metadata_sha256', 'actual_init_args'}}
        setup.update(alias_repair=proof, fresh_geometry='pending', fresh_demo='pending', fresh_replay='pending', production_released=False)
        candidate = replace(template, template_id=template.template_id + '__native_floating_r2',
            definition_sha256=digest(dict(entry=entry, bddl_sha256=bddl_sha, alias_repair=proof)), scene_setup=setup)
        check_reservations(candidate, reservations, hashes)
        native = GroundedProblem.load(source_job['alias'], str(customs), str(directory / 'bddl'))
        require(native.rooms(candidate.scene) == list(candidate.scene_setup.get('task_rooms', [candidate.room])), 'native candidate room changed')
        argv[argv.index('--customs') + 1] = str(customs.resolve()); argv[argv.index('--bddl-root') + 1] = str((directory / 'bddl').resolve())
        argv[argv.index('--out') + 1] = '${CAMPAIGN_OUT}/validation/' + candidate.template_id
        job = {**source_job, 'candidate_id': candidate.template_id, 'template_fingerprint': candidate.fingerprint,
            'source_manifest': str((a.output / 'validation_jobs.json').resolve()), 'dependency_release': str(current_release),
            'dependency_library': {**dep, 'alias': route['alias'], 'library_sha256': route['library_sha256'], 'alias_repair_proof': proof},
            'required_new_grasp_key': route['alias'], 'sample_argv': argv, 'status': 'native_alias_repaired_fresh_geometry_pending'}
        jobs.append(job); templates.append({**asdict(candidate), 'fingerprint': candidate.fingerprint})
    for filename, value in [('validation_jobs.json', {'jobs': jobs}), ('contract_candidates.json', {'templates': templates}),
        ('alias_repair_report.json', dict(unchanged_candidates=unchanged, changed_candidates=len(jobs), alias_audit_sha256=sha(a.alias_audit), active_policy_sha256=pointer['policy_sha256']))]:
        (a.output / filename).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(changed=len(jobs), unchanged=len(unchanged), manifest_sha256=sha(a.output / 'validation_jobs.json'))))


if __name__ == '__main__':
    main()
