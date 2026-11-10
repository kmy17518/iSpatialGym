"""Prepare four measured pair replacements; the allocator adopts separately."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True); stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    a = parser.parse_args(); root = a.root.resolve()
    from ispatialgym.grasping import schema as S
    from ispatialgym.sampling import grasp_library as G
    import os
    pointer = json.loads((root / 'manifests/active_reservations.json').read_text())
    old = Path(pointer['reservations']['path']); old_hash = root / 'manifests/reservation_revision6/reservation_asset_hashes.json'
    assert old.parent.name == 'reservation_revision6' and sha(old) == pointer['reservations']['sha256']
    base = json.loads(old.read_text()); hashes = json.loads(old_hash.read_text())
    assert hashes['reservation_sha256'] == sha(old)
    native = root / 'heldout_probes/ready_pair_replacements_screen_v1/default_geometry.json'
    visual = native.with_name('visual_review.json')
    doc = json.loads(native.read_text()); review = json.loads(visual.read_text())
    assert doc['status'] == 'native_defaults_and_geometry_recorded' and review['status'] == 'passed'
    assert review['default_geometry_sha256'] == sha(native)
    measured = {r['category']+'/'+r['model']: r for r in doc['objects']}
    changes = {
        'inter_object_relation_satisfaction-touching_02': ('target', 'box_of_cereal/pfwgkm'),
        'inter_object_relation_satisfaction-ontop_small_scale_04': ('partner', 'cutting_board/jfrbuz'),
        'inter_object_relation_satisfaction-under_small_scale_02': ('target', 'beer_glass/rxscji'),
        'inter_object_relation_satisfaction-under_small_scale_04': ('target', 'spatula/iktnwh'),
    }
    new_keys = sorted(key for _, key in changes.values())
    assert not set(new_keys) & set(base['train_exclude_models'])
    for key in new_keys:
        row = measured[key]
        assert row['ready'] and row['readiness_source'] and row['default_scale'] == [1., 1., 1.]
        assert sha(row['asset_path']) == row['asset_sha256'] == review['objects'][key]['asset_sha256']
        assert review['objects'][key]['status'] == 'passed'
        for image in review['objects'][key]['images']:
            assert sha(image['path']) == image['sha256']
    source = root / 'ready_pair_replacement_target_glib_v1'
    manifest = root / 'manifests/ready_pair_replacement_target_glib_v1.json'
    jobs = json.loads(manifest.read_text()); admissions = []
    os.environ['ISPATIALGYM_GRASP_LIBRARY_V2'] = str(source / 'glib'); G._v2_index.cache_clear()
    for index, job in enumerate(jobs):
        key = job['category']+'/'+job['model']; assert key in new_keys
        log = source / 'logs' / ('glib_' + str(index) + '.log')
        assert 'GB_VALIDATE_DONE' in log.read_text()
        proof = source / 'glib/logs' / ('validate_' + str(index) + 'of3.jsonl')
        events = [json.loads(line) for line in proof.read_text().splitlines() if line.strip()]
        assert len(events) == 1 and events[0]['ok'] and events[0]['n_valid'] > 0 and events[0]['id'] == job['id']
        lib = S.load_model(job['category'], job['model'], job['scale'], str(source/'glib'), constraint='tabletop', rest_hint_wxyz=None)
        assert lib and len(lib.grasps()) > 0 and Path(lib.path).parent.name == job['id']
        assert lib.data['dataset'] == job['dataset'] and lib.data['constraint'] == 'tabletop'
        S.check_compatible(lib, scale=measured[key]['default_scale'], rest_kind='stable')
        routed = G.v2_entry(job['legacy_key']); assert routed and Path(routed.path) == Path(lib.path)
        assert len(G.grasps_from_v2(routed, job['legacy_key'])) > 0
        admissions.append(dict(model_key=key, positive_grasps=len(lib.grasps()), library=str(lib.path), library_sha256=sha(lib.path),
            native_validation=str(proof), native_validation_sha256=sha(proof), validation_log=str(log), validation_log_sha256=sha(log)))
    assert {r['model_key'] for r in admissions} == set(new_keys)-{'cutting_board/jfrbuz'}
    policy = copy.deepcopy(base)
    for field in ('model_holdout_models','train_exclude_models'):
        policy[field] = sorted(set(policy[field]) | set(new_keys))
    replaced = []; seen = set()
    for template in policy['model_holdout_templates']:
        if template['class_id'] not in changes: continue
        role_name, key = changes[template['class_id']]; row = measured[key]
        matching = [role for role in template['roles'] if role['role'] == role_name]
        assert len(matching) == 1
        role = matching[0]; previous = role['category']+'/'+role['model']; replaced.append(dict(class_id=template['class_id'], role=role_name, previous=previous, new=key)); seen.add(template['class_id'])
        role.update(category=row['category'], model=row['model'], dataset=row['dataset'], ready=True,
            readiness_source=row['readiness_source'], asset_sha256=row['asset_sha256'], default_scale=row['default_scale'],
            geometry_validated=False, replaced_assignment=previous+' remains globally reserved')
    assert seen == set(changes)
    for key in new_keys: policy['readiness_evidence'][key] = measured[key]['readiness_source']
    policy['policy_extension'] = dict(revision=7, parent_reservation_sha256=sha(old), added_model_keys=new_keys, status='requires_allocator_adoption')
    out = root/'manifests/reservation_revision7_pairs'; assert not out.exists(); out.mkdir()
    new = out/'split_reservations.json'; save(new, policy)
    hashes['reservation_sha256'] = sha(new); hashes['reserved_model_keys'] = sorted(set(hashes['reserved_model_keys']) | set(new_keys))
    for key in new_keys:
        row = measured[key]; hashes['ready_assets'][key] = dict(asset_sha256=row['asset_sha256'],path=row['asset_path'],ready=True,readiness_provenance=row['readiness_source'])
    hashes['reserved_asset_hashes'] = sorted(set(hashes['reserved_asset_hashes']) | {measured[key]['asset_sha256'] for key in new_keys})
    new_hash = out/'reservation_asset_hashes.json'; save(new_hash, hashes)
    save(out/'extension_audit.json', dict(schema='campaign-monotonic-reservation-extension-v1',old_reservation_sha256=sha(old),new_reservation_sha256=sha(new),
        old_asset_hashes_sha256=sha(old_hash),new_asset_hashes_sha256=sha(new_hash),added_model_keys=new_keys,
        added_asset_sha256={key:measured[key]['asset_sha256'] for key in new_keys},added_categories=[],added_rooms=[],
        reason='Repair native pair size/aspect failures with measured ready default-scale models; retain every existing reservation.',
        native_defaults=dict(path=str(native),sha256=sha(native)),visual_review=dict(path=str(visual),sha256=sha(visual)),
        readiness_source={key:measured[key]['readiness_source'] for key in new_keys},target_grasp_admissions=admissions,
        changed_candidate_assignments=replaced,status='proposal_requires_locked_accepted_and_active_queued_use_audit_before_adoption'))
    for field in ('category_holdout_categories','category_holdout_models','train_exclude_categories','category_holdout_rooms'):
        assert base[field] == policy[field]
    for field in ('model_holdout_models','train_exclude_models'):
        assert set(base[field]) <= set(policy[field])
    print(json.dumps(dict(directory=str(out),reservation_sha256=sha(new),asset_hashes_sha256=sha(new_hash),audit_sha256=sha(out/'extension_audit.json'),admitted=admissions)))


if __name__ == '__main__': main()
