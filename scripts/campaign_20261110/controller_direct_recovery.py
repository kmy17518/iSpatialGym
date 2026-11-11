"""Claim only explicitly listed, terminal direct recovery groups after fresh replay.

This watcher submits no work. Direct groups do not execute a goal queue, so their
immutable allocation's terminal state supplies closure; native reports and the
ordinary acceptance auditor still establish every successful distinct member.
"""
import argparse
import json
import time
from pathlib import Path

from scripts.campaign_20261110 import controller_held as held
from scripts.campaign_20261110.acceptance import audit_job
from scripts.campaign_20261110.contracts import file_digest, require, write_json_atomic
from scripts.campaign_20261110.controller_families import slurm_status
from scripts.campaign_20261110.goal_groups import read_env
from scripts.campaign_20261110.group_lane import claim_group
from scripts.campaign_20261110.reservation_policy import active_policy
from scripts.campaign_20261110.storage import atomic_directory_lock


def read(path):
    return json.loads(Path(path).read_text())


def resolve_receipt(root, spec):
    path = root / 'dispatch/receipts' / (spec['token'] + '.json')
    chain = []
    while path.exists():
        require(path.resolve().is_relative_to(root / 'dispatch/receipts'), 'receipt outside campaign')
        require(str(path) not in [row['path'] for row in chain], 'receipt migration cycle')
        receipt = read(path)
        chain.append({'path': str(path), 'sha256': file_digest(path)})
        migration = receipt.get('migration')
        if migration:
            require(migration.get('never_started') is True, 'migration lacks never-started proof')
            proof = migration['proof']
            proof_path = Path(proof['path']).resolve()
            require(proof_path.is_relative_to(root / 'dispatch/migrations') and
                    file_digest(proof_path) == proof['sha256'], 'migration proof differs')
            path = Path(migration['replacement']['dispatch_receipt'])
            continue
        return receipt, chain
    return None, chain


def refresh(root, spec, record, cfg, queue):
    if record.get('status') == 'accepted':
        return
    receipt, chain = resolve_receipt(root, spec)
    record['receipt_chain'] = chain
    if not receipt or not receipt.get('allocation'):
        record['status'] = 'awaiting_allocation'
        return
    allocation = receipt['allocation']
    require(allocation['key'] == spec['class_id'] and int(allocation['n']) == 3, 'allocation scope differs')
    snapshot = Path(allocation['code']).resolve()
    require(snapshot.is_relative_to(root / 'code') and file_digest(snapshot / '.snapshot_commit') ==
            spec['validation_key']['code_sha256'], 'allocation runtime differs')
    command = receipt['routed_command']
    configs = [item.split('=', 1)[1] for item in command if item.startswith('ISPATIALGYM_CAMPAIGN_CONFIG=')]
    plans = [item.split('=', 1)[1] for item in command if item.startswith('ISPATIALGYM_CAMPAIGN_GOAL_PLAN=')]
    require(len(configs) == len(plans) == 1, 'expected one direct goal configuration')
    config_path, plan_path = Path(configs[0]).resolve(), Path(plans[0]).resolve()
    require(all(path.is_relative_to(root / 'groups') for path in (config_path, plan_path)), 'group input outside campaign')
    config, plan = read(config_path), read(plan_path)
    require(config['validation_key'] == spec['validation_key'] and config['split'] == 'eval_standard' and
            config['template']['class_id'] == spec['class_id'], 'direct contract differs')
    require(plan['split'] == 'eval_standard' and plan.get('source_instance_is_member') is False,
            'direct plan cannot reuse source instances')
    group_id = plan['group_id']
    record.update(allocation=allocation, group_id=group_id, configuration=str(config_path),
                  configuration_sha256=file_digest(config_path), goal_plan=str(plan_path),
                  goal_plan_sha256=file_digest(plan_path))
    if not held.job_terminal(allocation, root, queue):
        record['status'] = 'collecting'
        return
    job = Path(allocation['job_dir']).resolve()
    require(job.is_relative_to(root / 'out'), 'direct output outside campaign')
    reports = [read(path) for path in (job / 'demo').glob('*/meta.json')]
    record['demo_successes'] = sum(report.get('success') is True for report in reports)
    record['demo_failures'] = sum(report.get('success') is False for report in reports)
    record['native_written'] = len(list((job / 'inst').rglob('*-tro_state.json')))
    record['terminal_observed_at'] = time.time()
    if not record['demo_successes']:
        record.update(status='failed', reason='terminal_group_has_no_successful_demo')
        return
    gate_path = root / 'validation/open_loop' / allocation['key'] / allocation['jobid'] / 'gate.json'
    if not gate_path.exists():
        record.update(status='awaiting_replay', reason='own_fresh_physics_gate_missing')
        return
    gate = read(gate_path)
    require(Path(gate['source_job']).resolve() == job, 'direct recovery must use its own fresh replay gate')
    if gate.get('success') is not True:
        record.update(status='failed', reason='own_fresh_physics_replay_failed',
                      failed_gate={'path':str(gate_path), 'sha256':file_digest(gate_path)})
        return
    env = read_env(job / 'job.env')
    require(Path(env['ISGF_ROOT']).resolve() == root, 'job environment campaign differs')
    tro_groups = {read(path)['ispatialgym'].get('campaign', {}).get('goal_group_id')
                  for path in (job / 'inst').rglob('*-tro_state.json')}
    require(tro_groups == {group_id}, 'direct allocation contains another goal group')
    review = audit_job(job, gate_path, root=root, started_at=cfg['started_at'], group_closed=True, group_id=group_id)
    require(review['validation_key'] == spec['validation_key'], 'review key differs from listed recovery')
    template = held.template_from(config['template'])
    policy = active_policy(root)
    require(policy is not None, 'active holdout reservations unavailable')
    review_path = config_path.parent / 'release_review.json'
    write_json_atomic(review_path, review)
    result = claim_group(review, template, policy[1], reservation_hashes=policy[2], root=root,
                         started_at=cfg['started_at'], state_path=root / 'dataset/allocations.sqlite')
    record.update(status='accepted', accepted=result['accepted'], review=str(review_path),
                  review_sha256=file_digest(review_path), accepted_at=time.time())
    record.pop('reason', None)


def tick(config_path):
    cfg = read(config_path)
    root = Path(cfg['root']).resolve()
    require(root.name == 'campaign_20261110' and cfg['campaign_id'] == root.name, 'wrong campaign')
    specs = cfg['groups']
    require(len({spec['token'] for spec in specs}) == len(specs), 'duplicate recovery token')
    require(all('/' not in spec['token'] and spec['token'].startswith('standard_recovery_') for spec in specs),
            'unscoped recovery token')
    directory = root / 'controllers/standard_direct_recovery_templates'
    directory.mkdir(exist_ok=True)
    state_path = directory / 'state.json'
    with atomic_directory_lock(directory / 'state.lockdir'):
        state = read(state_path) if state_path.exists() else {'records': {}}
        queue = slurm_status()
        for spec in specs:
            record = state['records'].setdefault(spec['token'], {'class_id': spec['class_id']})
            try:
                refresh(root, spec, record, cfg, queue)
                record.pop('error', None)
            except Exception as exc:
                record['error'] = {'at': time.time(), 'reason': type(exc).__name__ + ': ' + str(exc)}
            record['observed_at'] = time.time()
            write_json_atomic(state_path, state)
        state['observed_at'] = time.time()
        write_json_atomic(state_path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        result = tick(args.config)
        print(json.dumps({token:{k:row[k] for k in ('status','accepted','reason','error') if k in row}
                          for token,row in result['records'].items()}), flush=True)
        if not args.watch:
            break
        time.sleep(30)


if __name__ == '__main__':
    main()
