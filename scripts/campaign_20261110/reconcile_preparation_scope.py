"""Publish an additive consumed-data projection; never dispatch or delete jobs."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from scripts.campaign_20261110.preparation_consumers import consumers, use_kind


def ref(path):
    path = Path(path)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def project(graph, scales):
    configurations = {c['configuration_id']: c for c in scales['rows']}
    retained, retired = [], []
    links = defaultdict(list)
    for row in graph['rows']:
        groups = defaultdict(list)
        for cid in row['consumers']:
            c = configurations.get(cid)
            scope = use_kind(c['class_id'], row['kind'], row['inputs']) if c else None
            if scope:
                groups[scope].append(cid)
        if not groups:
            retired.append({'id': row['id'], 'kind': row['kind'], 'reason': 'No selected code consumer under owner preparation scope',
                            'historical_status': row['status'], 'artifact': row.get('artifact')})
            continue
        value = {**row, 'consumers': sorted(cid for ids in groups.values() for cid in ids),
                 'consumer_scopes': dict(groups), 'status_observation': 'inherited_source_evidence_not_a_fresh_job_observation'}
        value['required_by_classes'] = sorted({configurations[c]['class_id'] for c in value['consumers']})
        retained.append(value)
        for scope, ids in groups.items():
            for cid in ids:
                links[cid].append({'id': row['id'], 'kind': row['kind'], 'scope': scope, 'source_status': row['status']})
    by_kind = defaultdict(Counter)
    for row in retained:
        by_kind[row['kind']][row['status']] += 1
    rows = []
    for cid, c in configurations.items():
        contract = consumers(c['class_id'])
        existing_kinds = {r['kind'] for r in links[cid]}
        rows.append({'configuration_id': cid, 'class_id': c['class_id'], 'variant_id': c.get('variant_id'),
                     'support_id': c.get('support_id'), 'is_canonical_eval_template': c.get('is_canonical_eval_template', False),
                     'dependencies': links[cid], 'consumer_kinds_not_in_inherited_graph': sorted(contract.required - existing_kinds),
                     'exact_scales_resolved': c['exact_scales_resolved'],
                     'per_scene_feasibility_required': False,
                     'no_v2_data_required': not contract.required and not contract.optional})
    contracts = {}
    for cls in sorted({c['class_id'] for c in configurations.values()}):
        value = asdict(consumers(cls))
        value['required'], value['optional'] = sorted(value['required']), sorted(value['optional'])
        contracts[cls] = value
    return {'rows': retained, 'configuration_requirements': rows, 'retired_requirements': retired, 'consumer_contracts': contracts,
            'summary': {'selected_configuration_choices': len(rows), 'retained_inherited_logical_requirements': len(retained),
                        'retired_inherited_logical_requirements': len(retired), 'retired_by_kind': dict(Counter(r['kind'] for r in retired)),
                        'retained_by_kind_and_source_status': {k: dict(v) for k, v in by_kind.items()},
                        'per_scene_feasibility_required': False, 'camera_v2_dependencies_required': 0,
                        'floating_maps_required': 0, 'legacy_reach_bands_required': 0,
                        'full_dependency_denominator_complete': False,
                        'physical_dedup_reconciliation_complete': False,
                        'current_ready_unsubmitted_jobs': None},
            'gaps': ['Join newly verified physical libraries/maps without reusing historical job liveness.',
                     'Expand actual customs-bound lean/overhang/interior/floor/tray/attachment/prefix data where the inherited graph was generic or absent.',
                     'Resolve generic maps by robot/config/height/arm/pose_set, including native object-height-derived dz; remove target/scene labels from physical dedup.',
                     'Retain all exact canonical100 inputs and map their existing release closures; no extra validation campaign.',
                     'Resolve remaining exact scale/height/interior/contact metadata only when consumed by selected data builders.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('graph', 'scales', 'guide', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    result = project(json.loads(a.graph.read_text()), json.loads(a.scales.read_text()))
    result.update(schema='isg-consumed-preparation-scope-v1', observed_at=time.time(), owner='/root/templates_assets',
                  source_graph=ref(a.graph), source_scales=ref(a.scales), guide=ref(a.guide), helper=ref(__file__),
                  consumer_policy=ref(Path(__file__).with_name('preparation_consumers.py')),
                  production_authorized=False, benchmark_credit=0,
                  authority='Owner 2026-11-12 scope correction: build only data consumed by code; no per-scene feasibility campaign')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, sort_keys=True, separators=(',', ':')) + '\n')
    print(json.dumps({'report': ref(a.output), 'summary': result['summary']}))


if __name__ == '__main__':
    main()
