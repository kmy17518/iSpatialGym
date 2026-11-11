"""Bind a finite ready standard pool without changing native task geometry.

The resulting configuration has a new validation key and is only a candidate.
It needs fresh native sampling, a successful demo and true physics replay.
"""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path

from scripts.campaign_20261110.binding_pools import BindingPool, PoolCandidate, bind_variant, write_pool
from scripts.campaign_20261110.contracts import digest, file_digest, require, write_json_atomic
from scripts.campaign_20261110.controller_held import template_from
from scripts.campaign_20261110.group_lane import check_reservations
from scripts.campaign_20261110.reservation_policy import active_policy


def make_pool(template, definition, *, maximum=48, provenance=None):
    require(template.task == 'disembedding' and template.split == 'eval_standard', 'standard disembedding source required')
    require(type(maximum) is int and 1 <= maximum <= 48, 'invalid finite pool maximum')
    roles = definition['roles']
    choices, selection = {}, {}
    for obj in template.objects:
        if obj.role not in ('target', 'distractor'):
            continue
        value = asdict(obj); value.pop('instance')
        key = 'model_' + digest(value)[:20]
        choices.setdefault(key, obj); selection[obj.instance] = key
    for role, declared_role in [('target', 'target'), ('distractor', 'distractors')]:
        declared = roles[declared_role]['category']
        declared = {declared} if isinstance(declared, str) else set(declared)
        actual = {obj.category for obj in choices.values() if obj.role == role}
        require(actual == declared, 'ready contract does not cover the complete declared ' + role + ' category pool')
    fixed_layout = bool(definition.get('generation', {}).get('fixed_layout'))
    counts = Counter(selection.values())
    n_distractors = sum(obj.role == 'distractor' for obj in template.objects)
    require(1 <= n_distractors <= maximum, 'active distractors exceed finite maximum')
    candidates = tuple(PoolCandidate(key, obj, counts[key] if fixed_layout else 1 if obj.role == 'target' else maximum)
                       for key, obj in sorted(choices.items()))
    identity = digest({'class': template.class_id, 'definition': template.definition_sha256,
                       'scene': template.scene, 'room': template.room, 'candidates': [asdict(c) for c in candidates]})
    pool = BindingPool('standard_disembedding_pool_' + identity[:20], template.class_id, template.split,
                       template.scene, template.room, tuple(o for o in template.objects if o.role == 'support'),
                       candidates, {'target': (1, 1), 'distractor': (n_distractors, n_distractors) if fixed_layout else (1, maximum)},
                       template.definition_sha256, tuple(template.scene_setup.get('task_rooms', [template.room])),
                       fixed_layout=fixed_layout, fixed_selection=selection if fixed_layout else {}, provenance=provenance or {})
    pool.check_selection(selection)
    return pool, selection


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('campaign-root', 'config', 'customs', 'snapshot', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--max-distractors', type=int, default=48)
    a = p.parse_args(); root = a.campaign_root.resolve(); out = a.out.resolve()
    require(out.is_relative_to(root) and not out.exists(), 'new owned output directory required')
    config = json.loads(a.config.read_text()); template = template_from(config['template'])
    require('binding_pool' not in template.scene_setup, 'source already has an immutable pool; preserve it')
    customs = json.loads(a.customs.read_text()); alias = template.scene_setup.get('customs_alias', template.scene_setup['source_alias'])
    require(alias in customs, 'source alias absent from exact native customs')
    active = active_policy(root); require(active is not None, 'active reservation policy required')
    proof = {'source_config': str(a.config.resolve()), 'source_config_sha256': file_digest(a.config),
             'source_customs': str(a.customs.resolve()), 'source_customs_sha256': file_digest(a.customs),
             'reservation_policy_sha256': active[0]['policy_sha256'],
             'geometry_models_counts_and_definition_unchanged': True,
             'scope': 'All models in the ready standard contract; every declared target/distractor category covered; finite copy maximum'}
    pool, selection = make_pool(template, customs[alias], maximum=a.max_distractors, provenance=proof)
    check_reservations(pool.audit_template(), active[1], active[2])
    out.mkdir(parents=True); bound = bind_variant(template, write_pool(out/'binding_pool.json', pool), selection)
    check_reservations(bound, active[1], active[2])
    key = {**config['validation_key'], 'template_sha256': bound.fingerprint,
           'code_sha256': file_digest(a.snapshot/'.snapshot_commit')}
    write_json_atomic(out/'campaign_config.json', {**config, 'purpose': 'validation', 'template': asdict(bound), 'validation_key': key})
    write_json_atomic(out/'pool_binding_proof.json', {**proof, 'pool_sha256': pool.fingerprint,
        'original_template_sha256': template.fingerprint, 'new_template_sha256': bound.fingerprint,
        'native_runtime': str(a.snapshot.resolve()), 'native_runtime_identity_sha256': key['code_sha256'],
        'fresh_native_demo_and_replay_required': True, 'accepted': False})
    print(json.dumps({'config': str(out/'campaign_config.json'), 'pool': str(out/'binding_pool.json'), 'class_id': template.class_id}))


if __name__ == '__main__':
    main()
