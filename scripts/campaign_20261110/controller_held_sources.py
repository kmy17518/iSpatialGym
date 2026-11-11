"""Admit validated held-object sources to explicitly owned evaluation refill."""
from __future__ import annotations

from scripts.campaign_20261110 import controller_lifecycle as lifecycle

import argparse
from pathlib import Path
import time

from scripts.campaign_20261110 import controller_held as held
from scripts.campaign_20261110.controller_admission import append_seed, observe_gate, recent_gates, source_in_scope, source_proof
from scripts.campaign_20261110.controller_families import allocations, read, slurm_status
from scripts.campaign_20261110.contracts import digest, require, write_json_atomic
from scripts.campaign_20261110.goal_groups import read_env
from scripts.campaign_20261110.storage import atomic_directory_lock


def source_split(cfg):
    split = cfg.get('source_split', 'eval_model')
    require(split in ('eval_model', 'eval_category'), 'held source requires model or category evaluation')
    if split == 'eval_category':
        tasks = cfg.get('admitted_tasks')
        require(isinstance(tasks, list) and tasks and len(set(tasks)) == len(tasks)
                and all(task in held.RANGES for task in tasks), 'category source requires explicit held task ownership')
    return split


def owner_key(spec, cfg):
    if source_split(cfg) == 'eval_category':
        require(spec.get('task') in cfg['admitted_tasks'], 'category seed outside owned tasks')
        return spec['task']
    return spec['class_id']


def make_seed(proof, allocation, cfg, root):
    template = held.template_from(proof['template'])
    split = source_split(cfg)
    require(template.split == split and template.task in held.RANGES,
            'held source requires model evaluation' if split == 'eval_model' else 'held source requires category evaluation')
    standards = read(cfg['standard_contracts'])['templates']
    if split == 'eval_model':
        excluded = cfg.get('excluded_classes')
        require(isinstance(excluded, list) and all(isinstance(value, str) for value in excluded), 'missing immutable model ownership exclusions')
        require(template.class_id not in excluded, 'model class is owned by another refill controller')
        require(any(row['class_id'] == template.class_id and row['task'] == template.task for row in standards),
                'held source class outside frozen standard inventory')
    else:
        require(template.task in cfg['admitted_tasks'], 'category source outside owned tasks')
        require(any(row['task'] == template.task for row in standards), 'category source task outside frozen standard inventory')
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
    result = {'id': ('held_model_' if split == 'eval_model' else 'held_category_')+identity[:20], 'class_id': template.class_id,
        'seed_config': proof['config']['path'], 'seed_config_sha256': proof['config']['sha256'],
        'source_gate': str(gate), 'source_gate_sha256': proof['source_gate']['sha256'],
        'anchor_job': str(job), 'anchor_ledger': allocation['ledger'], 'snapshot': str(snapshot),
        'v2_root': str(library.parent), 'max_candidates': 180, 'groups_per_batch': 1,
        'seed_offset': int(identity[:8], 16), 'group_budget_seconds': 900, 'existing_queues': []}
    if split == 'eval_category':
        result['task'] = template.task
    return result


def load_seeds(cfg, state):
    root = Path(cfg['root']).resolve()
    path = Path(cfg['seed_catalog']).resolve()
    require(path.is_relative_to(root/'controllers'), 'held catalog outside campaign')
    raw = read(path)
    require(raw['schema'] == 'isg-group-seed-catalog-v1' and raw['campaign_id'] == cfg['campaign_id'], 'wrong held catalog')
    specs = cfg['seeds'] + raw['seeds']
    require(len({spec['id'] for spec in specs}) == len(specs), 'duplicate held catalog ID')
    if source_split(cfg) == 'eval_category':
        require(len({owner_key(spec, cfg) for spec in specs}) == len(specs), 'category task has multiple pinned sources')
    seen = state.setdefault('admitted_seed_hashes', {})
    current = {spec['id']: digest(spec) for spec in raw['seeds']}
    require(all(current.get(key) == value for key, value in seen.items()), 'held seed removed or changed')
    from scripts.campaign_20261110.controller_source_allocations import SourceAllocations
    sources = SourceAllocations(root, state, allocations)
    for spec in raw['seeds']:
        if spec['id'] in seen:
            continue
        proof = source_proof(spec['source_gate'], root=root, started_at=cfg['started_at'],
                             campaign_id=cfg['campaign_id'], permitted_split=source_split(cfg))
        found = [sources.find(proof['source_job'], ledger=spec['anchor_ledger'])]
        require(make_seed(proof, found[0], cfg, root) == spec, 'held seed differs from fresh audited source')
        seen[spec['id']] = current[spec['id']]
    return specs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--state', required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args(); cfg = read(args.config); root = Path(cfg['root']).resolve(); path = Path(args.state).resolve()
    split = source_split(cfg)
    schema = 'isg-held-model-source-controller-v1' if split == 'eval_model' else 'isg-held-category-source-controller-v1'
    require(cfg['schema'] == schema and path.is_relative_to(root/'controllers'), 'invalid held source controller')
    while lifecycle.continue_running(cfg):
        with atomic_directory_lock(path.with_suffix('.lockdir')):
            state = read(path, {'config_sha256': digest(cfg), 'checked_gates': {}})
            require(state['config_sha256'] == digest(cfg), 'held source controller config changed')
            lifecycle.record_policy(cfg, state)
            catalog = read(cfg['seed_catalog']); owned = {owner_key(row, cfg) for row in catalog['seeds'] + cfg.get('seeds', [])}
            from scripts.campaign_20261110.controller_source_allocations import SourceAllocations
            sources, queue = SourceAllocations(root, state, allocations), slurm_status()
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
                if not source_in_scope(observed[1], root, {**cfg, 'admission_split': split}):
                    continue
                examined += 1
                try:
                    proof = source_proof(gate, root=root, started_at=cfg['started_at'], campaign_id=cfg['campaign_id'], permitted_split=split)
                    require(proof['template']['task'] in held.RANGES, 'source is not a held-object task')
                    found = [sources.find(proof['source_job'])]
                    if not held.job_terminal(found[0], root, queue):
                        continue
                    spec = make_seed(proof, found[0], cfg, root)
                    ownership = owner_key(spec, cfg)
                    if ownership not in owned:
                        append_seed(cfg['seed_catalog'], spec, cfg['campaign_id']); owned.add(ownership)
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
