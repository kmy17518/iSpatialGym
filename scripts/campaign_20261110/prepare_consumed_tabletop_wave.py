"""Stream missing simple-IORS tabletop maps from native positive libraries."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main(root, runtime, graph_path, old_inventory, output):
    from ispatialgym.grasping import schema as S
    from scripts.campaign_20261110.prepare_consumed_attachment_maps import prepare
    root, runtime, graph_path, old_inventory, output = map(Path, (root, runtime, graph_path, old_inventory, output))
    output.mkdir(parents=True, exist_ok=True); digests = {}; libraries = {}
    def sha(path):
        key = str(path)
        if key not in digests: digests[key] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return digests[key]
    ref = lambda p: {'path': str(p), 'sha256': sha(p)}
    read = lambda p: json.loads(Path(p).read_text())
    def load(reference, entry=None):
        path = Path(reference['path']); assert sha(path) == reference['sha256']
        if str(path) not in libraries:
            value = read(path); lib = S.load_model(value['category'], value['model'], value['scale'], str(path.parents[2]), constraint=value['constraint'])
            assert lib is not None and Path(lib.path) == path and lib.grasps()
            S.check_compatible(lib, scale=value['scale'])
            libraries[str(path)] = lib
        return libraries[str(path)]
    graph = read(graph_path); previous = read(old_inventory); plans = {}; owned = []; errors = []; unresolved_entries = set()
    for row in previous['rows']:
        try:
            source = row['source_plan']; assert sha(source['path']) == source['sha256']
            if source['path'] not in plans: plans[source['path']] = read(source['path'])
            plan = plans[source['path']]
            assert plan['runtime'] == str(runtime) or plan['generator_compatibility']['native_runtime'] == str(runtime)
            for name, digest in plan['generator_compatibility']['module_hashes'].items(): assert sha(runtime / name) == digest
            unit = next(u for u in plan['units'] if u['id'] == row['unit'])
            lib = load(unit['source_library'])
            owned.append({'library_identity': lib.identity_key(), 'library': unit['source_library'], 'height': unit['height'],
                          'arms': unit['arms'], 'source_plan': source, 'wave_unit_id': row['wave_unit_id'],
                          'disposition': 'Preserve original ownership, outputs and recovery; never duplicate this exact computation.'})
        except (OSError, ValueError, KeyError, AssertionError, StopIteration) as exc:
            unresolved_entries.add(row['entry_id']); errors.append({'wave_unit_id': row['wave_unit_id'], 'entry_id': row['entry_id'],
                'blocker': type(exc).__name__ + ': ' + str(exc), 'next_action': 'Reconcile existing ownership under its account; do not duplicate.'})
    ownership = output / 'owned_units.json'; ownership.write_text(json.dumps({'schema': 'isg-consumed-tabletop-existing-ownership-v1',
        'observed_at': time.time(), 'source_inventory': ref(old_inventory), 'units': owned, 'unresolved': errors}, indent=2) + '\n')
    rows = []; blocked = []; seen = set()
    for dep in graph['rows']:
        if dep['kind'] != 'tabletop_grasp_library' or dep['status'] != 'verified': continue
        if not all(c.startswith('inter_object_relation_satisfaction-') and 'large_scale' not in c for c in dep['required_by_classes']): continue
        source = dep['artifact']; path = Path(source['path'])
        if str(path) in seen: continue
        seen.add(str(path))
        try:
            if path.parent.name in unresolved_entries: raise ValueError('Existing same-entry ownership is unresolved')
            lib = load(source); value = lib.data; target = dep['inputs']['target']
            assert lib.scale == S.canonical_scale(target['applied_scale']) and lib.category == target['category'] and lib.model == target['model']
            if value['constraint'] != 'tabletop': raise ValueError('Nonstandard constraint needs its own bound consumer manifest')
            assert lib.rest_kind == 'stable' and all(lib.rests[g['rest_id']].stable for g in lib.grasps())
            files = [ref(p) for p in sorted(path.parent.iterdir()) if p.is_file() and p.suffix in {'.json', '.obj', '.npz'}]
            entry = {'id': path.parent.name, 'kind': 'tabletop', 'constraint': value['constraint'], 'category': lib.category,
                     'model': lib.model, 'scale': list(lib.scale), 'dataset': value.get('dataset'), 'ready': True,
                     'source': 'exact_selected_native_positive_library', 'campaign_classes': dep['required_by_classes']}
            rows.append({'status': 'verified_positive', 'native_reader_passed': True, 'positive_grasps': len(lib.grasps()),
                         'native_identity_key': lib.identity_key(), 'library': source, 'entry': entry, 'files': files,
                         'historical_native_evidence': dep.get('evidence', []), 'source_dependency_id': dep['id']})
        except (OSError, ValueError, KeyError, AssertionError) as exc:
            blocked.append({'dependency_id': dep['id'], 'library': source, 'blocker': type(exc).__name__ + ': ' + str(exc)})
    audit = output / 'library_native_reader.json'; audit.write_text(json.dumps({'schema': 'isg-consumed-tabletop-native-reader-v1',
        'observed_at': time.time(), 'owner': '/root/templates_assets', 'source_graph': ref(graph_path), 'rows': rows,
        'blocked': blocked, 'benchmark_credit': 0, 'production_authorized': False}, indent=2) + '\n')
    print(json.dumps({'native_positive_libraries': len(rows), 'blocked_libraries': len(blocked), 'old_physical_units_preserved': len(owned),
                      'old_ownership_unresolved': len(errors), 'audit': ref(audit)}), flush=True)
    prepare(root, runtime, graph_path, audit, output / 'prepared', family='tabletop', owned_units=ownership, chunk_size=64)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'runtime', 'graph', 'old-inventory', 'output'): parser.add_argument('--' + name, required=True)
    args = parser.parse_args(); main(args.root, args.runtime, args.graph, args.old_inventory, args.output)
