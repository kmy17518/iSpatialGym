"""Admit newly validated held-model sources to strict, bounded evaluation refill."""
from __future__ import annotations

from scripts.campaign_20261110 import controller_lifecycle as lifecycle

import argparse
from pathlib import Path
import time

from scripts.campaign_20261110 import controller_held as held
from scripts.campaign_20261110.controller_admission import append_seed, observe_gate, recent_gates, source_proof
from scripts.campaign_20261110.controller_families import allocations, read, slurm_status
from scripts.campaign_20261110.contracts import digest, require, write_json_atomic
from scripts.campaign_20261110.goal_groups import read_env
from scripts.campaign_20261110.storage import atomic_directory_lock


def make_seed(proof, allocation, cfg, root):
    template = held.template_from(proof['template'])
    require(template.split == 'eval_model' and template.task in held.RANGES, 'held source requires model evaluation')
    excluded = cfg.get('excluded_classes')
    require(isinstance(excluded, list) and all(isinstance(value, str) for value in excluded), 'missing immutable model ownership exclusions')
    require(template.class_id not in excluded, 'model class is owned by another refill controller')
    standards = read(cfg['standard_contracts'])['templates']
    require(any(row['class_id'] == template.class_id and row['task'] == template.task for row in standards),
            'held source class outside frozen standard inventory')
    require(all(proof['validation_key'][key] == cfg[key] for key in ('physics_sha256', 'data_sha256')),
            'held source differs from pinned campaign physics/data')
    job = Path(allocation['job_dir'])
    require(str(job.resolve()) == proof['source_job'], 'held source allocation differs')
    env = read_env(job/'job.env')
    held.source_environment(job, root=root)
    library = Path(env['ISPATIALGYM_GRASP_LIBRARY_V2']).resolve()
    maps = Path(env['ISPATIALGYM_REACH_MAPS']).resolve()
    require(library.name == 'glib' and maps.name == 'maps' and library.parent == maps.parent,
            'held source grasp/map release differs')
    require(library.is_dir() and maps.is_dir(), 'held source library release missing')
    gate = Path(proof['source_gate']['path'])
    snapshot = Path(read(gate)['snapshot']).resolve()
    identity = digest({'class': template.class_id, 'split': template.split, 'key': proof['validation_key']})
    return {'id': 'held_model_'+identity[:20], 'class_id': template.class_id,
        'seed_config': proof['config']['path'], 'seed_config_sha256': proof['config']['sha256'],
        'source_gate': str(gate), 'source_gate_sha256': proof['source_gate']['sha256'],
        'anchor_job': str(job), 'anchor_ledger': allocation['ledger'], 'snapshot': str(snapshot),
        'v2_root': str(library.parent), 'max_candidates': 180, 'groups_per_batch': 1,
        'seed_offset': int(identity[:8], 16), 'group_budget_seconds': 900, 'existing_queues': []}


def load_seeds(cfg, state):
    root = Path(cfg['root']).resolve()
    path = Path(cfg['seed_catalog']).resolve()
    require(path.is_relative_to(root/'controllers'), 'held catalog outside campaign')
    raw = read(path)
    require(raw['schema'] == 'isg-group-seed-catalog-v1' and raw['campaign_id'] == cfg['campaign_id'], 'wrong held catalog')
    specs = cfg['seeds'] + raw['seeds']
    require(len({spec['id'] for spec in specs}) == len(specs), 'duplicate held catalog ID')
    seen = state.setdefault('admitted_seed_hashes', {})
    current = {spec['id']: digest(spec) for spec in raw['seeds']}
    require(all(current.get(key) == value for key, value in seen.items()), 'held seed removed or changed')
    rows = allocations(root)
    for spec in raw['seeds']:
        if spec['id'] in seen:
            continue
        proof = source_proof(spec['source_gate'], root=root, started_at=cfg['started_at'],
                             campaign_id=cfg['campaign_id'], permitted_split='eval_model')
        found = [row for row in rows if Path(row['job_dir']).resolve() == Path(proof['source_job'])]
        require(len(found) == 1, 'held source allocation missing or ambiguous')
        require(make_seed(proof, found[0], cfg, root) == spec, 'held seed differs from fresh audited source')
        seen[spec['id']] = current[spec['id']]
    return specs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--state', required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args(); cfg = read(args.config); root = Path(cfg['root']).resolve(); path = Path(args.state).resolve()
    require(cfg['schema'] == 'isg-held-model-source-controller-v1' and path.is_relative_to(root/'controllers'), 'invalid held source controller')
    while lifecycle.continue_running(cfg):
        with atomic_directory_lock(path.with_suffix('.lockdir')):
            state = read(path, {'config_sha256': digest(cfg), 'checked_gates': {}})
            require(state['config_sha256'] == digest(cfg), 'held source controller config changed')
            lifecycle.record_policy(cfg, state)
            catalog = read(cfg['seed_catalog']); owned = {row['class_id'] for row in catalog['seeds'] + cfg.get('seeds', [])}
            rows, queue = allocations(root), slurm_status()
            examined = 0
            for gate in recent_gates(root):
                if examined >= 40:
                    break
                observed = observe_gate(gate)
                if observed is None or observed[1].get('success') is not True:
                    continue
                sha, _ = observed
                if state['checked_gates'].get(str(gate), {}).get('sha256') == sha:
                    continue
                examined += 1
                try:
                    proof = source_proof(gate, root=root, started_at=cfg['started_at'], campaign_id=cfg['campaign_id'], permitted_split='eval_model')
                    require(proof['template']['task'] in held.RANGES, 'source is not a held-object task')
                    found = [row for row in rows if Path(row['job_dir']).resolve() == Path(proof['source_job'])]
                    require(len(found) == 1, 'held source allocation missing or ambiguous')
                    if not held.job_terminal(found[0], root, queue):
                        continue
                    spec = make_seed(proof, found[0], cfg, root)
                    if spec['class_id'] not in owned:
                        append_seed(cfg['seed_catalog'], spec, cfg['campaign_id']); owned.add(spec['class_id'])
                    observation = {'sha256': sha, 'seed_id': spec['id'], 'class_id': spec['class_id'], 'status': 'validated_source_observed'}
                except Exception as exc:
                    observation = {'reason': type(exc).__name__+': '+str(exc)}
                    if not isinstance(exc, FileExistsError):
                        observation['sha256'] = sha
                state['checked_gates'][str(gate)] = {**observation, 'observed_at': time.time()}
                write_json_atomic(path, state)
            state['updated_at'] = time.time(); write_json_atomic(path, state)
        if args.once:
            break
        time.sleep(60)


if __name__ == '__main__':
    main()
