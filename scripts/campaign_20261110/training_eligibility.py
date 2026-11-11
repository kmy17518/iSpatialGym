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
    if allocations == {'train': 1} and len(members) == 1 and not group.get('deferred'):
        return {'status': 'eligible', 'category': 'eval_singleton', 'reasons': ['successful_standard_group_singleton_fallback']}
    return {'status': 'excluded', 'category': 'eval_overflow',
            'reasons': ['training_only_multi_member_group_has_no_standard_eval_purpose_remaining']}


_PROJECTION_CACHE = {}


def projection_records(root):
    path = Path(root)/'accounting/training_eligibility.json'
    if not path.exists(): return {}
    stat = path.stat()
    signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
    cached = _PROJECTION_CACHE.get(str(path))
    if cached and cached[0] == signature: return cached[1]
    value = json.loads(path.read_text())
    if value.get('campaign_id') != 'campaign_20261110' or value.get('status') != 'complete':
        raise ValueError('training eligibility projection unavailable')
    records = value.get('records', {})
    _PROJECTION_CACHE[str(path)] = (signature, records)
    return records


def eligible_training_records(db, task=None, *, root=None, class_id=None):
    """Quota basis: only proved eval siblings/singletons, never flagged capacity.

    Committed allocator records already carry their independently audited native,
    demo and replay provenance. The projection separately rechecks their bytes.
    """
    projection = projection_records(root) if root is not None else {}
    groups = {}
    query = "SELECT record FROM members WHERE split='train'"
    parameters = []
    if task: query += ' AND task=?'; parameters.append(task)
    if class_id: query += ' AND class_id=?'; parameters.append(class_id)
    result = []
    for value, in db.execute(query, parameters):
        row = json.loads(value)
        if row.get('sampling_split') != 'eval_standard' or row.get('split') != 'train': continue
        projected = projection.get(row.get('instance_id'))
        if projected and (projected.get('status') != 'eligible' or projected.get('accepted_record_sha256') != digest(row)): continue
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
    evidence_rows = row.get('evidence', [])
    if {item.get('kind') for item in evidence_rows} != {'demo', 'open_loop_replay'}:
        raise ValueError('missing exact demo/open-loop-replay evidence')
    for evidence in evidence_rows:
        if evidence.get('key') != row['validation_key'] or evidence.get('success') is not True or evidence.get('qa_passed') is not True:
            raise ValueError('validation evidence key/result mismatch')
        if evidence.get('restoration_verified') is not True or evidence.get('demo_sha256') != gate.get('demo_sha256'):
            raise ValueError('validation restoration/demo binding mismatch')
        physical = verify(evidence['report_path'], evidence['report_sha256'])
        report_passed(physical, 'oracle' if evidence['kind'] == 'demo' else 'replay')
        if physical.get('instance_id') != evidence.get('instance_id'):
            raise ValueError('validation report instance mismatch')
    return {'native_tro_sha256': row['tro_sha256'], 'demo_report_sha256': row['report_sha256'],
            'gate': row['gate'], 'validation_key': row['validation_key'],
            'room': row.get('room'), 'perturbations': block.get('perturbations')}


def standard_group_proof(root, class_id, validation_key):
    """Exact-key accepted eval+successful sibling proof, never a historical label."""
    import shutil
    import sqlite3
    import tempfile
    source = Path(root)/'dataset/allocations.sqlite'
    if not source.exists(): return None
    checked = set()
    with tempfile.TemporaryDirectory(prefix='isg-group-proof-', dir='/tmp') as temporary:
        local = Path(temporary)/'state.sqlite'
        shutil.copyfile(source, local)
        with sqlite3.connect(f'file:{local}?mode=ro', uri=True) as db:
            records = eligible_training_records(db, root=root, class_id=class_id)
            for row in records:
                if row.get('class_id') != class_id or row.get('validation_key') != validation_key: continue
                group_raw = db.execute('SELECT record FROM groups WHERE id=?', (row['group_id'],)).fetchone()
                group = json.loads(group_raw[0]) if group_raw else None
                if classify_record(row, group)['category'] != 'eval_sibling': continue
                try: audit_eligible(row, checked)
                except (OSError, ValueError, KeyError, TypeError): continue
                return {'group_id': row['group_id'], 'release_review_sha256': group['release_review_sha256'],
                        'validation_key': validation_key, 'gate': row['gate']}
    return None


def coverage_facts(row, raw, template=None):
    """Observed accepted provenance, with absent evidence distinct from zero.

    A scene's original furniture removal is not a randomized training removal.
    Model/scale values come from native bindings and restored observations; role
    labels are used only from the matching hash-bound campaign contract.
    """
    block = raw.get('campaign', {})
    scene = raw.get('scene', {})
    bindings = raw.get('object_bindings', {})
    runtime = block.get('runtime', {})
    perturb = block.get('perturbations', {})
    roles = {}
    if template is not None:
        if digest(template) != row['validation_key']['template_sha256']:
            raise ValueError('coverage contract differs from accepted template')
        roles = {obj['instance']: obj for obj in template['objects']}
        for name, obj in roles.items():
            native = bindings.get(name, {})
            if native.get('model') != obj['model']:
                raise ValueError('coverage contract/native model mismatch')
    models, supports, scales = [], [], []
    for name, native in bindings.items():
        category, model = native.get('category'), native.get('model')
        if not category or not model: continue
        obj = roles.get(name, {})
        if obj.get('semantic_kind') in ('floor', 'wall') or category in ('floors', 'walls'): continue
        models.append(category+'/'+model)
        if obj.get('role') == 'support':
            supports.append('/'.join((str(scene.get('model')), category, model, str(native.get('scene_object_name')))))
        default = perturb.get('scales', {}).get(name, {}).get('default')
        observed = runtime.get('observed_scales', {}).get(name)
        if default and observed and len(default) == len(observed) == 3 and all(float(x)>0 for x in default):
            ratios = [round(float(a)/float(b), 6) for a,b in zip(observed,default)]
            scales.append({'model': category+'/'+model, 'ratio': ratios,
                           'nondefault': any(abs(v-1)>1e-5 for v in ratios)})
    original_removed = scene.get('scene_variant') or block.get('scene_setup', {}).get('scene_variant') or {}
    original_removed = original_removed.get('removed_objects', [])
    requested_removed = perturb.get('removed_objects')
    moved = runtime.get('augmentation', {}).get('moved_objects')
    lighting = runtime.get('augmentation', {}).get('lighting')
    room = row.get('room')
    if not room and template is not None and template.get('room') in scene.get('task_rooms', []):
        room = str(scene.get('model'))+'/'+template['room']
    return {'scene': scene.get('model'), 'room': room, 'models': sorted(set(models)),
        'supports': sorted(set(supports)) if template is not None else None,
        'scales': scales, 'scale_objects_without_observation': len(models)-len(scales),
        'lighting': lighting, 'lighting_requested': perturb.get('lighting'),
        'original_scene_removed_objects': sorted(original_removed),
        'augmentation_removed_objects': sorted(requested_removed) if isinstance(requested_removed, list) else None,
        'augmentation_moved_objects': sorted(moved) if isinstance(moved, dict) else None}


def coverage_summary(facts):
    counts = {name: Counter() for name in ('scenes', 'rooms', 'models', 'supports', 'scale_ratios', 'lighting_states')}
    missing = Counter()
    measures = Counter()
    for fact in facts:
        if fact.get('error'):
            missing['record'] += 1
            continue
        for name, field in (('scenes','scene'), ('rooms','room')):
            if fact.get(field): counts[name][fact[field]] += 1
            else: missing[name] += 1
        for name in ('models', 'supports'):
            if fact.get(name) is None: missing[name] += 1
            else: counts[name].update(set(fact[name]))
        missing['scale_objects'] += fact.get('scale_objects_without_observation', 0)
        for scale in fact['scales']:
            counts['scale_ratios'][json.dumps(scale['ratio'], separators=(',', ':'))] += 1
        measures['instances_with_nondefault_scale'] += any(s['nondefault'] for s in fact['scales'])
        lighting = fact.get('lighting')
        if lighting is None: missing['lighting'] += 1
        else:
            counts['lighting_states'][digest(lighting)] += 1
            measures['instances_with_lighting_augmentation'] += bool(lighting)
            measures['instances_with_no_lighting_augmentation'] += not lighting
        measures['instances_with_original_scene_removal'] += bool(fact['original_scene_removed_objects'])
        removed, moved = fact.get('augmentation_removed_objects'), fact.get('augmentation_moved_objects')
        if removed is None or moved is None: missing['augmentation_removal'] += 1
        else:
            measures['instances_with_verified_augmentation_removal'] += bool(removed) and set(removed).issubset(moved)
            measures['instances_with_no_augmentation_removal'] += not removed
            if removed and not set(removed).issubset(moved): missing['augmentation_removal_restoration'] += 1
    return {'basis': 'eligible accepted instances only; native observations and hash-bound object roles',
        'eligible_instances': len(facts), 'coverage_evidence_complete': not any(missing.values()),
        'missing_evidence': dict(missing), 'measures': dict(measures),
        **{name: {'distinct': len(values), 'counts': dict(sorted(values.items()))} for name,values in counts.items()},
        'note': 'Counts measure observed coverage, not distribution compliance. Original scene removal is separate from training augmentation; {} lighting means no recorded augmentation.'}


def build_projection(root):
    from scripts.campaign_20261110.export_release import snapshot_allocations
    root = Path(root)
    rows, groups, inputs = snapshot_allocations(root)
    now = datetime.now(timezone.utc).isoformat()
    cohorts, records, checked = {}, {}, set()
    coverage, templates = [], {}
    coverage_tasks, coverage_classes = defaultdict(list), defaultdict(list)
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
            try:
                gid = row.get('group_id')
                if gid not in templates:
                    path = root/'groups'/gid/'campaign_config.json'
                    templates[gid] = json.loads(path.read_text())['template'] if path.is_file() else None
                native = json.loads(Path(row['tro']).read_text())['ispatialgym']
                fact = coverage_facts(row, native, templates[gid])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                fact = {'error': str(exc)}
            coverage.append(fact)
            coverage_tasks[row['task']].append(fact)
            coverage_classes[row.get('class_id', row['task'])].append(fact)
            proof['coverage'] = fact
        key = (row['task'], row.get('class_id', row['task']), decision['status'], decision['category'], tuple(decision['reasons']))
        cohort_id = digest(key)[:20]
        cohort = cohorts.setdefault(cohort_id, {'cohort_id': cohort_id, 'task': key[0], 'class_id': key[1],
            'status': decision['status'], 'category': decision['category'], 'reasons': decision['reasons'],
            'count': 0, 'remediation': 'retain original; validate compliant replacement or concrete missing evidence',
            'coverage_evidence': [], 'instance_ids': []})
        cohort['count'] += 1
        cohort['instance_ids'].append(row['instance_id'])
        if proof and len(cohort['coverage_evidence']) < 3: cohort['coverage_evidence'].append(proof)
        records[row['instance_id']] = {**decision, 'cohort_id': cohort_id, 'accepted_record_sha256': digest({k:v for k,v in row.items() if k != 'allocation_source'}),
            'allocation_source': row['allocation_source'], 'group_id': row.get('group_id'), 'evidence': proof}
    from scripts.campaign_20261110.contracts import task_quotas
    for q in task_quotas(): tasks.setdefault(q.task, 0)
    for row in rows: classes.setdefault(row.get('class_id', row['task']), 0)
    return {'schema': 'isg-training-eligibility-v1', 'campaign_id': 'campaign_20261110', 'observed_at': now,
        'status': 'complete', 'coverage_complete': True, 'historically_accepted': sum(totals.values()), 'eligible_accepted': totals['eligible'],
        'excluded': totals['excluded'], 'unverified': totals['unverified'], 'replacement_count': 0,
        'remaining': 100000-totals['eligible'], 'replacement_links': [], 'inputs': inputs,
        'tasks': [{'task': task, 'eligible_accepted': n} for task,n in sorted(tasks.items())],
        'classes': [{'class_id': cls, 'eligible_accepted': n} for cls,n in sorted(classes.items())],
        'coverage': {**coverage_summary(coverage), 'excluded_historical_instances_not_counted': totals['excluded'],
                     'unverified_instances_not_counted': totals['unverified'],
                     'by_task': {task:coverage_summary(coverage_tasks[task]) for task in sorted(tasks)},
                     'by_class': {cls:coverage_summary(coverage_classes[cls]) for cls in sorted(classes)}},
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
