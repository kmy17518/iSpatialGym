"""Evidence-backed training eligibility without rewriting historical acceptance."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from scripts.campaign_20261110.contracts import digest, file_digest, write_json_atomic


def classify_record(row, group=None):
    """Eligibility category from committed goal-group allocation, not volume."""
    if row.get('split') != 'train':
        return {'status': 'not_training', 'category': row.get('split'), 'reasons': []}
    source = row.get('sampling_split')
    if source == 'train':
        return {'status': 'excluded', 'category': 'standalone_training',
                'reasons': ['complete_training_pipeline_not_validated', 'distribution_compliance_not_established']}
    if source != 'eval_standard':
        return {'status': 'unverified', 'category': 'unverified', 'reasons': ['sampling_purpose_unverified']}
    if row.get('task', '').startswith('camera_relocalization'):
        return {'status': 'excluded', 'category': 'camera_eval_overflow',
                'reasons': ['camera_exempt_from_goal_group_sibling_and_singleton_requirement', 'standalone_distribution_unverified']}
    if not group or group.get('accepted', {}).get(row.get('instance_id')) != 'train':
        return {'status': 'unverified', 'category': 'unverified', 'reasons': ['committed_group_assignment_missing']}
    allocations = Counter(group['accepted'].values())
    members = group.get('members', [])
    if not any(m.get('instance_id') == row.get('instance_id') and digest(m) == digest({k:v for k,v in row.items() if k != 'allocation_source'}) for m in members):
        return {'status': 'unverified', 'category': 'unverified', 'reasons': ['group_member_record_mismatch']}
    if allocations.get('eval_standard') == 1 and 1 <= allocations.get('train', 0) <= 2 and sum(allocations.values()) <= 3:
        return {'status': 'eligible', 'category': 'eval_sibling', 'reasons': ['committed_standard_eval_with_training_sibling']}
    if allocations == {'train': 1} and len(members) == 1:
        return {'status': 'eligible', 'category': 'eval_singleton', 'reasons': ['successful_standard_group_singleton_fallback']}
    return {'status': 'excluded', 'category': 'eval_overflow',
            'reasons': ['training_only_multi_member_group_has_no_standard_eval_purpose_remaining']}


def eligible_training_records(db, task=None):
    """Quota basis: only proved eval siblings/singletons, never flagged capacity.

    Committed allocator records already carry their independently audited native,
    demo and replay provenance. The projection separately rechecks their bytes.
    """
    groups = {}
    query = "SELECT record FROM members WHERE split='train'" + (' AND task=?' if task else '')
    result = []
    for value, in db.execute(query, (task,) if task else ()):
        row = json.loads(value)
        if row.get('sampling_split') != 'eval_standard' or row.get('split') != 'train': continue
        gid = row.get('group_id')
        if gid not in groups:
            found = db.execute('SELECT record FROM groups WHERE id=?', (gid,)).fetchone()
            groups[gid] = json.loads(found[0]) if found else None
        if classify_record(row, groups[gid])['status'] == 'eligible': result.append(row)
    return result


def audit_eligible(row, checked):
    from scripts.campaign_20261110.acceptance import report_passed
    def verify(path, expected):
        key = (path, expected)
        if key not in checked:
            if file_digest(path) != expected: raise ValueError('accepted evidence hash changed: ' + path)
            checked.add(key)
        return json.loads(Path(path).read_text())
    raw = verify(row['tro'], row['tro_sha256'])['ispatialgym']
    block = raw['campaign']
    if raw['instance_id'] != row['instance_id'] or block['split'] != 'eval_standard' or block['validation_key'] != row['validation_key']:
        raise ValueError('accepted native provenance mismatch')
    report = verify(row['report'], row['report_sha256'])
    report_passed(report, 'oracle')
    gate = verify(row['gate']['path'], row['gate']['sha256'])
    if gate.get('success') is not True or gate.get('replay_mode') != 'physics_action_open_loop':
        raise ValueError('accepted native replay gate failed')
    for evidence in row.get('evidence', []):
        verify(evidence['report_path'], evidence['report_sha256'])
    return {'native_tro_sha256': row['tro_sha256'], 'demo_report_sha256': row['report_sha256'],
            'gate': row['gate'], 'validation_key': row['validation_key'],
            'room': row.get('room'), 'perturbations': block.get('perturbations')}


def build_projection(root):
    from scripts.campaign_20261110.export_release import snapshot_allocations
    root = Path(root)
    rows, groups, inputs = snapshot_allocations(root)
    now = datetime.now(timezone.utc).isoformat()
    cohorts, records, checked = {}, {}, set()
    tasks, classes, totals = Counter(), Counter(), Counter()
    for row in rows:
        if row['split'] != 'train': continue
        decision = classify_record(row, groups.get(row.get('group_id')))
        proof = None
        if decision['status'] == 'eligible':
            try: proof = audit_eligible(row, checked)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                decision = {'status': 'unverified', 'category': decision['category'], 'reasons': [str(exc)]}
        totals[decision['status']] += 1
        if decision['status'] == 'eligible':
            tasks[row['task']] += 1
            classes[row.get('class_id', row['task'])] += 1
        key = (row['task'], row.get('class_id', row['task']), decision['status'], decision['category'], tuple(decision['reasons']))
        cohort_id = digest(key)[:20]
        cohort = cohorts.setdefault(cohort_id, {'cohort_id': cohort_id, 'task': key[0], 'class_id': key[1],
            'status': decision['status'], 'category': decision['category'], 'reasons': decision['reasons'],
            'count': 0, 'remediation': 'retain original; validate compliant replacement or concrete missing evidence',
            'coverage_evidence': [], 'instance_ids': []})
        cohort['count'] += 1
        cohort['instance_ids'].append(row['instance_id'])
        if proof and len(cohort['coverage_evidence']) < 3: cohort['coverage_evidence'].append(proof)
        records[row['instance_id']] = {**decision, 'cohort_id': cohort_id, 'accepted_record_sha256': digest(row),
            'allocation_source': row['allocation_source'], 'group_id': row.get('group_id'), 'evidence': proof}
    return {'schema': 'isg-training-eligibility-v1', 'campaign_id': 'campaign_20261110', 'observed_at': now,
        'status': 'complete', 'historically_accepted': sum(totals.values()), 'eligible_accepted': totals['eligible'],
        'excluded': totals['excluded'], 'unverified': totals['unverified'], 'replacement_count': 0,
        'remaining': 100000-totals['eligible'], 'replacement_links': [], 'inputs': inputs,
        'tasks': [{'task': task, 'eligible_accepted': n} for task,n in sorted(tasks.items())],
        'classes': [{'class_id': cls, 'eligible_accepted': n} for cls,n in sorted(classes.items())],
        'cohorts': list(cohorts.values()), 'records': records,
        'raw_generated': None, 'raw_generated_note': 'not inferred from accepted totals; see native attempt accounting',
        'policy': 'standalone blocked; preserve historical acceptance; eligible standard siblings/singletons only'}


def publish(root):
    root = Path(root)
    value = build_projection(root)
    directory = root/'accounting/training_eligibility_history'
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory/(str(time.time_ns())+'.json')
    write_json_atomic(archive, value)
    value['history_artifact'] = {'path': str(archive), 'sha256': file_digest(archive)}
    prior = root/'accounting/training_eligibility.json'
    history = []
    if prior.exists():
        old = json.loads(prior.read_text())
        history = old.get('history', [])[-999:]
        history.append({'observed_at': old['observed_at'], 'eligible_counts': {'train': old['eligible_accepted']}})
    value['history'] = history
    write_json_atomic(prior, value)
    return {k:value[k] for k in ('observed_at','historically_accepted','eligible_accepted','excluded','unverified')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.root)))
