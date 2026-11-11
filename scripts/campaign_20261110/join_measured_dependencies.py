"""Attribute executed support and initial-table map proofs to exact requests.

Native reader/asset/file checks happen while constructing the input evidence.
This join requires all selected identities to agree; it never fills layout,
tray-floor, interior, contact, articulated, or goal-context map requirements.
"""
from collections import Counter
from copy import deepcopy

from scripts.campaign_20261110.contracts import digest


def support_matches(selected, proof):
    p, binding = proof['probe'], proof['binding']
    return (p.get('geometry_admitted') is True and
            not p.get('conservative_overlap_blockers') and
            selected.get('scene') == p['scene'] and
            selected.get('object') == p['object'] and
            selected.get('height_slot') == p['slot'] and
            selected.get('support_profile') == p['profile_id'] and
            selected.get('selected_scale') == p['absolute_scale'] and
            selected.get('source_geometry_sha256') == binding['source_geometry_sha256'])


def map_matches(request, row):
    inputs = request['inputs']
    target, job = inputs.get('target') or {}, row['job']
    # These proofs are of the initial table plane. A tray/board/attachment
    # query cannot inherit one just because its base table has the same height.
    classes = request['required_by_classes']
    if request['kind'] != 'tabletop_reach_map' or not classes or not all(
            c.startswith('inter_object_relation_satisfaction-') for c in classes):
        return False
    return (row.get('status') == 'verified' and row['map'].get('native_reader_passed') is True and
            row['map'].get('positive_cells', 0) > 0 and
            all(target.get(k) == job.get(k) for k in ('category', 'model', 'dataset', 'asset_sha256')) and
            target.get('applied_scale') == job['scale'] and
            set(classes).issubset(row['consumer']['classes']) and
            inputs.get('arm') == row['map']['arm'] and
            row['map']['object_key'] == row['native_library_identity_key'] and
            row['map']['rest_id'] in row['positive_rest_ids'] and
            f"{row['map']['height']:.4f}" == row['support']['probe']['actual_height_key'] and
            support_matches(inputs.get('support') or {}, row['support']))


def join(manifest, evidence, evidence_ref):
    out = deepcopy(manifest)
    support_ids, map_ids = [], []
    for request in out['artifact_requests']:
        if request['kind'] == 'native_support_geometry_and_height':
            matches = [p for p in evidence['supports'] if support_matches(request['inputs'], p)]
            if not matches:
                continue
            proof = matches[0]
            artifact, identity = proof['probe_ref'], {
                'kind': request['kind'], 'probe': proof['probe_ref'], 'scene': proof['stable_scene']}
            resolved = {'measured_height_above_floor': proof['probe']['measured_height_above_floor'],
                        'actual_height_key': proof['probe']['actual_height_key'],
                        'stable_scene': proof['stable_scene'], 'layout_verified': False}
            support_ids.append(request['id'])
        elif request['kind'] == 'tabletop_reach_map':
            matches = [r for r in evidence['maps'] if map_matches(request, r)]
            if not matches:
                continue
            # All physically positive library rests must have a positive exact
            # map; selecting a single convenient rest would overstate closure.
            proof = matches[0]
            same = [r for r in matches if r['library'] == proof['library']]
            if {r['map']['rest_id'] for r in same} != set(proof['positive_rest_ids']):
                continue
            artifact = evidence_ref
            identity = {'kind': request['kind'], 'library': proof['library'],
                        'maps': [r['map'] for r in same], 'generator': proof['generator']}
            resolved = {'query_stage': 'initial_table_support_only',
                        'measured_height_above_floor': proof['support']['probe']['measured_height_above_floor'],
                        'actual_height_key': proof['support']['probe']['actual_height_key'],
                        'library': proof['library'], 'maps': [r['map'] for r in same],
                        'generator': proof['generator'], 'task_goal_query_verified': False}
            map_ids.append(request['id'])
        else:
            continue
        request.update(status='verified', input_resolution='resolved', unresolved_inputs=[],
                       physical_artifact_id='physical_' + digest(identity), artifact=artifact,
                       evidence=[evidence_ref], resolved_inputs=resolved, active_jobs=[],
                       next_action='This exact dependency is verified; layout, goal-query and task validation remain independent',
                       observed_at=evidence['observed_at'])
    by_id = {r['id']: r for r in out['artifact_requests']}
    for config in out['configurations']:
        required = [by_id[k] for k in config['dependencies']]
        config['verified_dependency_count'] = sum(r['status'] in ('verified', 'not_applicable') for r in required)
        config['remaining_dependency_count'] = len(required) - config['verified_dependency_count']
        if not config['remaining_dependency_count']:
            config['status'] = 'dependencies_verified_configuration_unmaterialized'
    for row in out['per_class']:
        configs = [c for c in out['configurations'] if c['class_id'] == row['class_id']]
        ids = {k for c in configs for k in c['dependencies']}
        row['dependency_statuses'] = dict(Counter(by_id[k]['status'] for k in ids))
        row['fully_served'] = sum(c['remaining_dependency_count'] == 0 for c in configs)
    out['summary']['dependency_statuses'] = dict(Counter(r['status'] for r in by_id.values()))
    out['summary']['resolved_physical_artifact_identities'] = sum(r['physical_artifact_id'] is not None for r in by_id.values())
    out['summary']['fully_served_configurations'] = sum(c['remaining_dependency_count'] == 0 for c in out['configurations'])
    out['measured_dependency_join'] = {'evidence': evidence_ref, 'support_request_ids': support_ids,
                                       'initial_table_map_request_ids': map_ids}
    return out
