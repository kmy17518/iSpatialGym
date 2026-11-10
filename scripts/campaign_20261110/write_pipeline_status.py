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
    a = p.parse_args(); root = a.root.resolve()
    now = datetime.now(timezone.utc).isoformat()
    rows = {}

    def put(task, cls, split, stage, reason, evidence, rank):
        key = (split, cls)
        files = [dict(path=str(path), sha256=sha(path)) for path in dict.fromkeys(evidence)]
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
               'heldout_block_materialized_v2', 'heldout_mental_isgm10_v1', 'heldout_lef_ief09_v1', 'heldout_pot_tbsjnw_v1', 'heldout_model_table_tasks_v1')
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
    bound_path = root / 'manifests/floating_consumer_contracts_r5.json'
    bound = json.loads(bound_path.read_text())
    for template in bound['templates']:
        if template['split'] in ('eval_model', 'eval_category'):
            put(template['task'], template['class_id'], template['split'], 'fresh_native_sample_bound',
                'Fresh native consumer output is bound to exact ready assets, scales and definition. Annotated demonstration/replay and allocation are tracked separately.', [bound_path], 4)
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
