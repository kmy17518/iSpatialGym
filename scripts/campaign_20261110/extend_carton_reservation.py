#!/usr/bin/env python3
"""Propose the measured compact carton; allocator must audit and adopt separately."""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    if path.exists():
        raise ValueError('immutable output exists')
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def main():
    inv = ROOT / 'tmp/campaign_20261110/inventory'
    out = inv / 'reservation_revision5'
    out.mkdir(parents=True, exist_ok=True)
    old = inv / 'reservation_revision4/split_reservations.json'
    old_hash = inv / 'reservation_revision4/reservation_asset_hashes.json'
    native = ROOT / 'tmp/campaign_20261110/remote_evidence/heldout_probes/compact_carton_candidates_v1/default_geometry.json'
    visual = native.parent / 'model_visual_review.json'
    doc, review = json.loads(native.read_text()), json.loads(visual.read_text())
    key = 'carton/hhlmbi'
    row = next(r for r in doc['objects'] if r['category'] + '/' + r['model'] == key)
    if doc['status'] != 'native_defaults_and_geometry_recorded' or not row['ready'] or review['default_geometry_sha256'] != sha(native) or review['objects'][key]['status'] != 'passed':
        raise ValueError('native readiness/visual proof missing')
    base, hashes = json.loads(old.read_text()), json.loads(old_hash.read_text())
    if hashes['reservation_sha256'] != sha(old) or key in base['train_exclude_models']:
        raise ValueError('base identity mismatch or already reserved')
    policy = copy.deepcopy(base)
    for field in ('model_holdout_models', 'train_exclude_models'):
        policy[field] = sorted(set(policy[field]) | {key})
    policy['readiness_evidence'][key] = row['readiness_source']
    changed = []
    for template in policy['model_holdout_templates']:
        for role in template['roles']:
            if role['role'] == 'target' and role.get('category') == 'carton' and role.get('model') == 'msfzpz':
                role.update(model='hhlmbi', dataset=row['dataset'], ready=True, readiness_source=row['readiness_source'],
                    asset_sha256=row['asset_sha256'], default_scale=row['default_scale'], geometry_validated=False,
                    native_joint='j_link0', replaced_assignment='carton/msfzpz remains globally reserved but unused')
                changed.append(template['class_id'])
    if len(changed) != 2:
        raise ValueError('expected two small carton task assignments')
    policy['policy_extension'] = dict(revision=5, parent_reservation_sha256=sha(old), added_model_keys=[key], status='requires_allocator_adoption',
        unchanged_policy='Existing reserved models, categories and rooms remain reserved.')
    new = out / 'split_reservations.json'
    save(new, policy)
    hashes['reservation_sha256'] = sha(new)
    hashes['reserved_model_keys'] = sorted(set(hashes['reserved_model_keys']) | {key})
    hashes['ready_assets'][key] = dict(asset_sha256=row['asset_sha256'], path=row['asset_path'], ready=True, readiness_provenance=row['readiness_source'])
    hashes['reserved_asset_hashes'] = sorted(set(hashes['reserved_asset_hashes']) | {row['asset_sha256']})
    new_hash = out / 'reservation_asset_hashes.json'
    save(new_hash, hashes)
    save(out / 'extension_audit.json', dict(schema='campaign-monotonic-reservation-extension-v1',
        old_reservation_sha256=sha(old), new_reservation_sha256=sha(new), old_asset_hashes_sha256=sha(old_hash), new_asset_hashes_sha256=sha(new_hash),
        added_model_keys=[key], added_asset_sha256={key: row['asset_sha256']}, added_categories=[], added_rooms=[],
        reason='Replace oversized default-scale small-articulation carton with native compact hinged carton; retain msfzpz globally reserved and unused.',
        readiness_source=row['readiness_source'], native_defaults_sha256=sha(native), visual_review_sha256=sha(visual),
        source_native_defaults='/vision/u/minyeong/projects/isg_home/campaign_20261110/heldout_probes/compact_carton_candidates_v1/default_geometry.json',
        changed_candidate_assignments=changed, old_model_stays_reserved='carton/msfzpz',
        status='proposal_requires_locked_accepted_contract_and_camera_audit_before_atomic_adoption'))
    for field in ('model_holdout_models', 'category_holdout_models', 'train_exclude_models'):
        if not set(base[field]) <= set(policy[field]):
            raise ValueError('not monotonic')
    for field in ('category_holdout_categories', 'train_exclude_categories', 'category_holdout_rooms', 'category_only_exempt_semantic_kinds', 'structural_model_category_exempt'):
        if base[field] != policy[field]:
            raise ValueError('existing category or room policy changed')
    print(json.dumps(dict(reservation_sha256=sha(new), hashes_sha256=sha(new_hash), extension_audit_sha256=sha(out / 'extension_audit.json'))))


if __name__ == '__main__':
    main()
