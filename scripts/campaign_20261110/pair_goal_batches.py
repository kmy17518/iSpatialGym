#!/usr/bin/env python3
"""Author fresh start groups at native pair goals in the exact validated scene."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random

from ispatialgym.sampling.campaign_goal_queue import PAIR_SCHEMA, canonical_pair_goal, read_queue
from ispatialgym.sampling.campaign_pair_goals import validate_pair_plan
from scripts.campaign_20261110 import goal_groups as g
from scripts.campaign_20261110.contracts import digest, file_digest, require, write_json_atomic
from scripts.campaign_20261110.goal_reuse_policy import POLICY as GOAL_REUSE_POLICY
from scripts.campaign_20261110.held_goal_batches import submission_command as submission_command  # Same bounded queue launch protocol.

__all__ = ["prepare_batch", "submission_command", "translate_pair_plan"]


def translate_pair_plan(raw, dx, dy):
    """Translate the complete small pair; native support/contact checks decide fit.

    Relative positions/orientations, touching pre-push geometry, scales, and the
    fixed scene remain identical. This is a goal proposal, never a start state.
    """
    require(raw['task'].endswith('_small_scale'), 'scene-bound large partners cannot be translated')
    require(all(math.isfinite(v) for v in (dx, dy)) and .01 <= math.hypot(dx, dy) <= .061,
            'pair goal proposal must move 1–6 cm')
    result = copy.deepcopy(raw)
    for key in ('goal_pose', 'release_pose', 'partner_pose', 'place_pose'):
        if key in result:
            result[key]['position'][0] += dx
            result[key]['position'][1] += dy
    return result


def prepare_batch(*, anchor, template, campaign_root, batch_id, count, seed, campaign_id,
                  code_sha256, physics_sha256, data_sha256, budget_seconds=900, goal_mode="anchor"):
    require(batch_id and all(c.isalnum() or c in '_.-' for c in batch_id), 'invalid batch ID')
    require(isinstance(count, int) and 1 <= count <= 32, 'pair batch requires 1–32 goal groups')
    require(goal_mode in ("anchor","native_region"), "unknown pair goal proposal mode")
    relation, scale, _ = template.task.split('-', 1)[1].split('_')
    require(template.task.startswith('inter_object_relation_satisfaction-') and
            (scale == 'small' or (scale == 'large' and relation == 'nextto')), 'unsupported pair goal proposal family')
    root = Path(campaign_root).resolve()
    queue_dir = root/'queues'/batch_id
    require(not queue_dir.exists(), 'queue already exists')
    rng = random.Random(seed)
    entries, groups = [], []
    for ordinal in range(count):
        group_id = f'{batch_id}_{ordinal:04d}'
        directory = root/'groups'/group_id
        g.prepare_group(anchor=anchor, template=template, campaign_id=campaign_id, group_id=group_id, output=directory,
                        code_sha256=code_sha256, physics_sha256=physics_sha256, data_sha256=data_sha256,
                        magnitude_range=[0, 1], budget_seconds=budget_seconds, max_members=3,
                        reuse_anchor_goal=goal_mode == "anchor")
        path = directory/'goal_plan.json'
        plan = g.read_json(path)
        proposal = {'method': 'fresh_anchor_exact_goal' if goal_mode == "anchor" else 'native_nextto_middle_eighty_percent',
                    'seed': seed, 'ordinal': ordinal, 'goal_reuse_policy':GOAL_REUSE_POLICY}
        if scale == 'small' and goal_mode == "native_region":
            radius, angle = rng.uniform(.015, .06), rng.uniform(-math.pi, math.pi)
            dx, dy = radius*math.cos(angle), radius*math.sin(angle)
            plan = translate_pair_plan(plan, dx, dy)
            proposal.update(method='rigid_pair_translation_before_starts', translation_m=[dx, dy, 0],
                            requires_native_support_geometry_contact_and_reach_checks=True)
        signature = digest(canonical_pair_goal(plan))
        stamp = plan['created_at']
        validate_pair_plan(plan, template_id=plan['template_id'], scene=template.scene, rooms=[template.room],
                           relation=relation, scale=scale, activated_at=stamp, now=stamp)
        write_json_atomic(path, plan)
        write_json_atomic(directory/'goal_proposal.json', {**proposal, 'anchor_sources': anchor['sources'],
                                                         'goal_sha256': signature})
        entries.append({'path': str(path), 'sha256': file_digest(path)})
        groups.append(group_id)
    config_path = root/'groups'/groups[0]/'campaign_config.json'
    require(all(file_digest(root/'groups'/name/'campaign_config.json') == file_digest(config_path) for name in groups),
            'queued pair groups differ in scene/scale/light/config identity')
    queue_dir.mkdir(parents=True, exist_ok=False)
    queue_path = queue_dir/'queue.json'
    write_json_atomic(queue_path, {'schema': PAIR_SCHEMA, 'campaign_id': campaign_id, 'config_path': str(config_path),
                                  'config_sha256': file_digest(config_path), 'entries': entries, 'group_ids': groups, 'seed': seed,
                                  'goal_reuse_policy':GOAL_REUSE_POLICY})
    read_queue(queue_path, config_path)
    return {'queue': str(queue_path), 'config': str(config_path), 'group_ids': groups, 'candidate_instances': 3*len(groups)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('anchor-job', 'campaign-root', 'anchor-ledger', 'contracts', 'class-id', 'campaign-id', 'batch-id',
                 'code-identity-file', 'physics-spec', 'data-manifest'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--campaign-started-at', type=float, required=True)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--budget-seconds', type=float, default=900)
    parser.add_argument('--goal-mode', choices=('anchor','native_region'), default='anchor')
    args = parser.parse_args()
    anchor = g.successful_anchor(args.anchor_job, campaign_root=args.campaign_root, ledger=args.anchor_ledger,
                                 campaign_started_at=args.campaign_started_at)
    print(json.dumps(prepare_batch(anchor=anchor, template=g.load_template_contract(args.contracts,args.class_id),
        campaign_root=args.campaign_root, batch_id=args.batch_id, count=args.count, seed=args.seed, campaign_id=args.campaign_id,
        code_sha256=file_digest(args.code_identity_file), physics_sha256=file_digest(args.physics_spec),
        data_sha256=file_digest(args.data_manifest), budget_seconds=args.budget_seconds, goal_mode=args.goal_mode)))


if __name__ == '__main__':
    main()
