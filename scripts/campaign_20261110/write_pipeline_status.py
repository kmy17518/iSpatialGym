#!/usr/bin/env python3
"""Report preparation evidence without modifying validation or acceptance counts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--secondary-root', type=Path)
    a = p.parse_args(); root = a.root.resolve()
    now = datetime.now(timezone.utc).isoformat()
    rows = {}

    def evidence_file(path):
        path = Path(path)
        digest = sha(path)
        if path.resolve().is_relative_to(root):
            return dict(path=str(path.resolve()), sha256=digest)
        secondary = a.secondary_root.resolve() if a.secondary_root else None
        assert secondary and path.resolve().is_relative_to(secondary), 'evidence outside owned campaign roots'
        # The dashboard only opens primary-root evidence. Preserve the exact
        # source bytes in a content-addressed primary copy; never change the
        # secondary source or its permissions.
        target = root / 'manifests/pipeline_evidence' / digest / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with target.open('xb') as stream:
                stream.write(path.read_bytes())
        assert sha(target) == digest and sha(path) == digest
        provenance = target.with_name(target.name + '.source.json')
        record = dict(source=str(path), source_resolved=str(path.resolve()), source_sha256=digest,
                      copy=str(target), copy_sha256=digest, source_permissions_unchanged=True)
        if not provenance.exists():
            with provenance.open('x') as stream:
                json.dump(record, stream, indent=2); stream.write('\n')
        assert json.loads(provenance.read_text()) == record
        return dict(path=str(target), sha256=digest, source_path=str(path),
                    provenance_path=str(provenance), provenance_sha256=sha(provenance))

    def put(task, cls, split, stage, reason, evidence, rank):
        key = (split, cls)
        files = [evidence_file(path) for path in dict.fromkeys(evidence)]
        previous = rows.get(key)
        if previous and previous[0] > rank:
            return
        rows[key] = (rank, dict(task=task, class_id=cls, split=split, stage=stage, reason=reason, observed_at=now, evidence=files))

    pointer = json.loads((root / 'manifests/active_reservations.json').read_text())
    # Resolve the committed pointer rather than assuming a particular revision.
    assert pointer['schema'] == 'isg-active-reservations-v1'
    reservation = Path(pointer['reservations']['path'])
    assert sha(reservation) == pointer['reservations']['sha256'], 'active reservation bytes differ'
    policy = json.loads(reservation.read_text())
    for proposal in policy['model_holdout_templates']:
        put(proposal['task'], proposal['class_id'], 'eval_model', 'ready_role_candidates_reserved',
            'Globally reserved role proposal exists; this report contains no physical task gate for this class.', [reservation], 0)
    default_path = root / 'heldout_probes/remaining_alignment_attachment_disembed_native_v2/default_geometry.json'
    review_path = default_path.with_name('model_visual_review_r2.json')
    if not review_path.is_file():
        review_path = default_path.with_name('model_visual_review.json')
    if default_path.is_file() and review_path.is_file():
        defaults, review = json.loads(default_path.read_text()), json.loads(review_path.read_text())
        if defaults.get('status') == 'native_defaults_and_geometry_recorded' and review.get('default_geometry_sha256') == sha(default_path):
            measured = {row['category'] + '/' + row['model']: row for row in defaults['objects']}
            for proposal in policy['model_holdout_templates']:
                actor_rows = [row for row in proposal['roles'] if row['role'] in ('target', 'partner', 'distractors', 'distractor')]
                if any('category' not in row or 'model' not in row for row in actor_rows):
                    continue
                actors = [row['category'] + '/' + row['model'] for row in actor_rows]
                if actors and all(key in measured for key in actors):
                    followups = [key for key in actors if review['objects'].get(key, {}).get('status') != 'passed']
                    put(proposal['task'], proposal['class_id'], 'eval_model',
                        'native_role_visual_followup' if followups else 'native_role_defaults_recorded',
                        ('Opposite-face/frame review still required: ' + ', '.join(followups)) if followups else
                        'Exact native default scales, bounds and collision meshes are recorded and appearance-reviewed; task geometry and dependencies remain separate gates.',
                        [default_path, review_path, reservation], 1)
    folders = ('heldout_room_materialized_v2', 'heldout_small_pairs_v1', 'heldout_articulation_v1', 'heldout_carton_hhlmbi_v1',
               'heldout_block_materialized_v2', 'heldout_mental_isgm10_v1', 'heldout_lef_ief09_v1', 'heldout_pot_tbsjnw_v1', 'heldout_model_table_tasks_v1',
               'heldout_model_disembedding_v1', 'heldout_model_wall_v3', 'heldout_small_pairs_replacements_v1',
               'held_touching02_replacement_fronts_v1', 'heldout_category_disembedding_reception_v1',
               'heldout_model_disembedding_garage_v1', 'heldout_model_disembedding_garage_right_v2',
               'heldout_model_disembedding_garage_ordered_v1', 'heldout_category_disembedding_reception_right_v1',
               'held_touching02_ordered_fronts_v1', 'heldout_wechsler_v1',
               'heldout_category_large_artic_open_state_v3')
    manifests = [root / folder / 'contract_candidates.json' for folder in folders]
    index_path = root / 'manifests/category_materialization_index_v4.json'
    index = json.loads(index_path.read_text())
    for item in index['manifests']:
        path = Path(item['contracts'])
        assert sha(path) == item['contracts_sha256'], 'category index hash differs'
        manifests.append(path)
    class_tasks = {}
    for path in manifests:
        if not path.is_file():
            continue
        for template in json.loads(path.read_text())['templates']:
            split, cls, task = template['split'], template['class_id'], template['task']
            class_tasks[(split, cls)] = task
            put(task, cls, split, 'candidate_materialized',
                'Exact private task definition and ready-object candidate are materialized. Enumeration is preparation only; native task geometry, demonstration and true replay are not implied.', [path], 2)
    audit_path = root / 'manifests/pair_library_admission_v4/audit.json'
    audit = json.loads(audit_path.read_text())
    for status in ('admitted', 'skipped'):
        for item in audit[status]:
            source = Path(item['source_manifest'])
            assert sha(source) == item['source_manifest_sha256'], 'grasp source manifest changed'
            source_job = json.loads(source.read_text())[item['source_index']]
            split = 'eval_model' if item['split_kind'] == 'model' else 'eval_category'
            for cls in source_job['campaign_classes']:
                task = class_tasks.get((split, cls))
                if not task:
                    continue
                put(task, cls, split, 'target_grasp_library_positive' if status == 'admitted' else 'blocked_target_grasp_library',
                    'Exact target library passed native identity/scale and positive-grasp admission. Task maps, task geometry, demo and replay require their own evidence.' if status == 'admitted' else
                    'Target library was excluded by native admission: ' + item['reason'] + '. No threshold or scale exception applied.', [audit_path, source], 3)
    dependency_path = root / 'manifests/held_dependency_admission_v1/audit.json'
    if dependency_path.is_file():
        dependency = json.loads(dependency_path.read_text())
        for item in dependency['admitted']:
            require_positive = item.get('positive_grasps', 0) > 0
            assert require_positive, 'admission report contains nonpositive library'
            # The digest-bound native admission report records the private
            # library audit; this status reporter does not reopen cross-account
            # library bytes or claim a later map/physical gate.
            for cls in item['campaign_classes']:
                task = class_tasks.get(('eval_model', cls))
                if task:
                    put(task, cls, 'eval_model', 'target_grasp_library_positive',
                        'Exact target library passed native identity, default-scale and positive-grasp admission. Exact-height maps and task geometry/demo/replay remain separate gates.',
                        [dependency_path], 3)
    native_batches = [(root, root / 'validation_inputs/wall_dispenser_corridor_v1/validation_jobs.json', 'held_wall_dispenser_corridor_v1'),
                      (root, root / 'validation_inputs/held_model_disembedding_geometry_v1/validation_jobs.json', 'held_model_disembedding_geometry_v1'),
                      (root, root / 'validation_inputs/held_small_pairs_reviewed_fronts_v1/validation_jobs.json', 'held_pairs_reviewed_fronts_v1'),
                      (root, root / 'validation_inputs/category_under_pair_geometry_v1/validation_jobs.json', 'category_under_pair_geometry_v1'),
                      (root, root / 'validation_inputs/held_touching_consumers_v1/validation_jobs.json', 'held_touching_consumers_v1'),
                      (root, root / 'validation_inputs/held_axial05_consumer_v1/validation_jobs.json', 'held_axial05_consumer_v1'),
                      (root, root / 'validation_inputs/held_attachment_consumers_v1/validation_jobs.json', 'held_attachment_consumers_v1'),
                      (root, root / 'validation_inputs/held_ontop04_replacement_geometry_v1/validation_jobs.json', 'held_ontop04_replacement_geometry_v1'),
                      (root, root / 'validation_inputs/held_under_replacements_geometry_v1/validation_jobs.json', 'held_under_replacements_geometry_v1'),
                      (root, root / 'validation_inputs/held_axial05_consumer_v2/validation_jobs.json', 'held_axial05_consumer_v2'),
                      (root, root / 'validation_inputs/held_axial05_consumer_v3/validation_jobs.json', 'held_axial05_consumer_v3'),
                      (root, root / 'validation_inputs/held_touching02_replacement_geometry_v1/validation_jobs.json', 'held_touching02_replacement_geometry_v1'),
                      (root, root / 'validation_inputs/held_touching02_replacement_geometry_v2/validation_jobs.json', 'held_touching02_replacement_geometry_v2'),
                      (root, root / 'validation_inputs/category_reception_disembedding_geometry_v1/validation_jobs.json', 'category_reception_disembedding_geometry_v1'),
                      (root, root / 'validation_inputs/model_garage_disembedding_geometry_v1/validation_jobs.json', 'model_garage_disembedding_geometry_v1'),
                      (root, root / 'validation_inputs/model_garage_disembedding_right_geometry_v1/validation_jobs.json', 'model_garage_disembedding_right_geometry_v1'),
                      (root, root / 'validation_inputs/model_garage_disembedding_ordered_geometry_v1/validation_jobs.json', 'model_garage_disembedding_ordered_geometry_v1'),
                      (root, root / 'validation_inputs/model_garage_disembedding_ordered_geometry_v1/validation_jobs.json', 'vast_native_ordered_model_disembedding04_r1'),
                      (root, root / 'validation_inputs/category_reception_disembedding_right_geometry_v1/validation_jobs.json', 'category_reception_disembedding_right_geometry_v1'),
                      (root, root / 'validation_inputs/held_touching02_ordered_geometry_v1/validation_jobs.json', 'held_touching02_ordered_geometry_v1')]
    if a.secondary_root:
        secondary = a.secondary_root.resolve()
        assert secondary.name == 'campaign_20261110', 'unexpected secondary campaign source'
        native_batches += [(secondary, secondary / 'heldout_model_wall_bootstrap_v1/validation_jobs.json', 'held_model_wall_bootstrap_v1'),
                           (secondary, secondary / 'pair_bootstrap_inputs_v1/validation_jobs.json', 'held_pairs_native_bootstrap_v1')]
    native_observations = []
    for source_root, jobs_path, tag in native_batches:
        if not jobs_path.is_file():
            continue
        for job in json.loads(jobs_path.read_text())['jobs']:
            directory = source_root / 'materialized_runs' / tag / 'validation' / job['class_id']
            summary_path = directory / 'summary.json'
            if not summary_path.is_file():
                continue
            native_observations.append((summary_path.stat().st_mtime_ns, str(summary_path), jobs_path, job))
    # Newest completed native evidence wins across primary and secondary roots.
    for _, summary_name, jobs_path, job in sorted(native_observations):
        summary_path = Path(summary_name)
        summary = json.loads(summary_path.read_text())
        if not summary.get('finished'):
            continue
        assert summary['args']['variant'] == job['alias'] and summary['args']['scene'] == job['scene'], 'native status identity mismatch'
        counts = {}
        for templates in summary.get('problems', {}).values():
            for record in templates.values():
                for reason, count in record.get('rejections', {}).items():
                    counts[reason] = counts.get(reason, 0) + count
        written = summary.get('n_written', 0)
        reason = (f'Native bounded sampler wrote {written} fresh sample(s); exact binding, demo and replay remain separate gates.' if written else
                  'Last completed bounded native attempt wrote no sample: ' + (', '.join(f'{key}={value}' for key, value in sorted(counts.items())) or 'no valid template choices; inspect native summary') + '. No thresholds changed.')
        put(job['task'], job['class_id'], job['split'], 'fresh_native_sample_written' if written else 'blocked_native_geometry',
            reason, [summary_path, jobs_path], 4)
    bound_paths = [root / 'manifests/floating_consumer_contracts_r5.json']
    for pattern in ('held_model_wall_fresh_contracts_r*.json', 'held_pairs_fresh_contracts_r*.json',
                    'category_ontop_fresh_contract_r*.json', 'held_nextto03_fresh_contract_r*.json',
                    'held_attachment_fresh_contract_r*.json', 'held_axial05_fresh_contract_r*.json',
                    'category_reception_disembedding_fresh_contract_r*.json',
                    'model_garage_disembedding_fresh_contract_r*.json',
                    'model_garage_disembedding_right_fresh_contract_r*.json',
                    'model_garage_disembedding_ordered_fresh_contract_r*.json',
                    'model_garage_disembedding_ordered_vast04_fresh_contract_r*.json',
                    'category_reception_disembedding_right_fresh_contract_r*.json',
                    'category_floating_fresh_contract_r*.json', 'category_jar_close_fresh_contract_r*.json',
                    'category_wardrobe_open_fresh_contract_v3_r*.json',
                    'held_touching02_ordered_fresh_contract_r*.json'):
        bound_paths.extend(sorted((root / 'manifests').glob(pattern)))
    for bound_path in bound_paths:
        bound = json.loads(bound_path.read_text())
        for template in bound['templates']:
            if template['split'] in ('eval_model', 'eval_category'):
                put(template['task'], template['class_id'], template['split'], 'fresh_native_sample_bound',
                    'Fresh native consumer output is bound to exact ready assets, scales and definition. Annotated demonstration/replay and allocation are tracked separately.', [bound_path], 5)
    output = dict(schema='isg-campaign-pipeline-status-v1', campaign_id='campaign_20261110',
                  rows=[value[1] for _, value in sorted(rows.items())])
    text = json.dumps(output, indent=2, sort_keys=True) + '\n'
    version = root / 'manifests' / ('pipeline_status_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json')
    assert not version.exists(); version.write_text(text)
    current = root / 'manifests/pipeline_status.json'
    temp = current.with_suffix('.json.new'); temp.write_text(text); temp.replace(current)
    print(json.dumps(dict(rows=len(rows), path=str(current), sha256=sha(current), versioned=str(version))))


if __name__ == '__main__':
    main()
