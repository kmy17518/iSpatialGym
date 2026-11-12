"""Enumerate exact inside grasp constraints from role and canonical metadata."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time


def inventory(root, runtime, graph_path, output):
    import yaml
    from ispatialgym.grasping import schema as S
    root, runtime, graph_path, output = map(Path, (root, runtime, graph_path, output))
    assert not output.exists()
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    ref = lambda p: {'path': str(p), 'sha256': sha(p)}
    read = lambda p: json.loads(Path(p).read_text())
    customs_path = runtime / 'ispatialgym/assets/task_specs/isg_customs.json'
    customs = read(customs_path); graph = read(graph_path); wanted = {}
    for row in graph['rows']:
        if row['kind'] != 'tabletop_grasp_library': continue
        classes = row['required_by_classes']
        if not all(c.startswith('inter_object_relation_satisfaction-inside_') and 'large_scale' not in c for c in classes): continue
        target = row['inputs']['target']; category = target['category']
        roles = [v['roles']['target'] for k, v in customs.items() if k.startswith('inter_object_relation_satisfaction-inside_small_scale-')
                 and isinstance(v, dict) and v.get('roles', {}).get('target', {}).get('category') == category]
        assert roles and all(r['require']['grasps'] == 'insert' for r in roles)
        canonical_keys = {key for r in roles for key in r['grasp_keys'].values()}
        assert len(canonical_keys) == 1
        canonical_key = canonical_keys.pop()
        legacy = runtime / 'runlogs/grasp_library_20260905'
        result_path = legacy / 'results' / (canonical_key + '.json'); result = read(result_path)
        candidates_path = legacy / 'cands' / (canonical_key + '.yaml')
        meta = (yaml.safe_load(candidates_path.read_text()) or {})['ispatialgym']
        constraint = result.get('constraint') or meta['constraint']
        assert constraint in {'tabletop_insert', 'flower_stem_top'}
        # A designated model rest may be reused at the selected scale of the
        # same model. New models measure their own canonical designated rest.
        same_model = result['model'] == target['model']
        hint = meta.get('rest_quat_wxyz') if same_model and meta.get('rest_pose', 'canonical') != 'canonical' else None
        scale = list(S.canonical_scale(target['applied_scale']))
        eid = S.entry_id(category, target['model'], scale, constraint, hint)
        job = {'id': eid, 'category': category, 'model': target['model'], 'dataset': target['dataset'], 'scale': scale,
               'constraint': constraint, 'kind': 'tabletop', 'rest_hint_wxyz': hint,
               'rest_pose': meta.get('rest_pose', 'canonical') if hint else 'canonical',
               'legacy_key': None, 'source': 'consumed_inside_exact_constraint_scale', 'asset_sha256': target['asset_sha256']}
        alias = category + '__' + target['model'] + '__selected_' + hashlib.sha256(eid.encode()).hexdigest()[:16]
        wanted[eid] = {'target': target, 'job': job, 'consumers': row['consumer_scopes']['required'], 'request_ids': [row['id']],
                       'source_refs': [ref(graph_path), ref(customs_path), ref(result_path), ref(candidates_path)],
                       'consumer_binding': {'required_constraint': constraint, 'consumer_key': alias, 'source_canonical_key': canonical_key,
                                            'registration_required': True, 'standard_hint_replacement_required': True},
                       'rest_policy': 'same_model_designated_rest' if hint else 'measure_own_canonical_rest'}
    output.mkdir(parents=True)
    (output / 'wanted.json').write_text(json.dumps(wanted, indent=2) + '\n')
    found, errors = [], []
    roots = [root / 'training_preparation', root / 'scale_prebuild', root.parent / 'oracle_fix/v2maps_int2'] + list(root.glob('v2_release*'))
    for base in roots:
        if not base.is_dir(): continue
        for directory, dirs, files in os.walk(base, onerror=lambda e: errors.append(str(e))):
            dirs[:] = [d for d in dirs if d not in {'sampling_cache', 'asset_cache', 'logs', 'native', 'native_readers', 'jobs', 'vast_routes_r1'}
                       and not d.startswith(('floating_', 'camera_'))]
            if Path(directory).name in wanted and 'library.json' in files:
                p = Path(directory) / 'library.json'; data = read(p); job = wanted[p.parent.name]['job']
                assert all(data[k] == job[k] for k in ('category', 'model', 'scale', 'constraint'))
                found.append({'entry': p.parent.name, 'library': ref(p), 'status': 'exact_candidate_requires_native_audit'})
    cache = {'observed_at': time.time(), 'roots': list(map(str, roots)), 'hits': found, 'errors': errors}
    (output / 'cache_inventory.json').write_text(json.dumps(cache, indent=2) + '\n')
    # Existing claims contain the exact prepared input. Inspect every consumed
    # build claim; old generic standard jobs are different constraints.
    matches = []; namespace = root / 'infra/coordination/preparation_dispatch/consumed_asset_builds_r1'
    for p in (namespace / 'receipts').glob('*.json'):
        value = read(p); ticket = value.get('ticket') or value
        prepared = ticket.get('prepared')
        if not isinstance(prepared, dict): continue
        try: doc = read(prepared['path'])
        except (PermissionError, FileNotFoundError): continue
        if doc.get('entry_id') in wanted:
            matches.append({'entry': doc['entry_id'], 'receipt': ref(p), 'prepared': prepared})
    (output / 'existing_claim_audit.json').write_text(json.dumps({'observed_at': time.time(), 'matches': matches}, indent=2) + '\n')
    (output / 'secondary_wanted.json').write_text(json.dumps({'schema': 'isg-specialized-consumer-wanted-v1', 'entry_ids': sorted(wanted),
        'primary_wanted': ref(output / 'wanted.json')}, indent=2) + '\n')
    result = {'observed_at': time.time(), 'wanted': ref(output / 'wanted.json'), 'count': len(wanted),
              'constraints': dict(Counter(r['job']['constraint'] for r in wanted.values())), 'cache_hits': len(found),
              'cache_errors': len(errors), 'existing_claim_matches': len(matches), 'benchmark_credit': 0}
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n'); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'runtime', 'graph-path', 'output'): p.add_argument('--' + name, required=True)
    a = p.parse_args(); inventory(a.root, a.runtime, a.graph_path, a.output)
