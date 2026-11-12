"""Apply exact source-verified scene scales, retaining immutable old identities.

This changes input resolution only. Changed query inputs lose all cached positive
status until native verification; unchanged physical evidence is retained.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_ref(path):
    return {'path': str(path), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def apply(expansion, proof, jobs, source_verification):
    if not source_verification.get('verified'):
        raise ValueError('exact scene sources have not been verified')
    proofs = {row['configuration_id']: row for row in proof['rows']}
    if len(proofs) != len(proof['rows']):
        raise ValueError('duplicate configuration evidence')
    original_jobs = {row['id']: row for row in jobs['rows']}
    result_rows, result_jobs, revisions, dependency_revisions = [], {}, [], []
    for original in expansion['rows']:
        row = deepcopy(original)
        edits = []
        for measured in proofs.get(row['configuration_id'], {}).get('roles', []):
            if measured['state'] != 'literal_exact_scene_scale_observed':
                continue
            source = measured['scene_source']
            checked = source_verification['sources'].get(source['path'], {})
            if checked.get('sha256') != source['sha256'] or (not measured.get('binding_unambiguous') and digest(measured) not in source_verification.get('identical_scale_roles', [])):
                raise ValueError('unverified scene identity or ambiguous binding')
            scale = measured['applied_scale']
            if len(scale) != 3 or not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in scale):
                raise ValueError('invalid measured scale')
            matches = [r for r in row['roles'] if all(r[k] == measured[k] for k in ('role', 'model_key', 'asset_sha256'))]
            if len(matches) != 1 or matches[0]['applied_scale'] is not None:
                raise ValueError('evidence does not identify exactly one unresolved role')
            matches[0].update(applied_scale=scale, scale_rule='literal_exact_scene_initialization', scene_scale_evidence=source,
                              scene_matching_instances=measured['matching_instances'])
            edits.append(measured)
            if not measured.get('binding_unambiguous'):
                row.setdefault('unresolved_scene_bindings', []).append(measured['role'])
        old_id = original['configuration_id']
        if edits:
            identity = {k: row.get(k) for k in ('class_id', 'scene', 'rooms', 'variant_id', 'support_id', 'roles',
                         'object_scale_slot', 'support_height_slot', 'selected_joint')}
            row['configuration_sha256'] = digest(identity)
            row['configuration_id'] = 'configuration_' + row['configuration_sha256'][:24]
            row['id'] = 'scale_choice_' + digest({'configuration': row['configuration_id'], 'roles': row['roles']})[:24]
            row['supersedes_configuration_id'] = old_id
            row['exact_scales_resolved'] = all(r['applied_scale'] is not None for r in row['roles'])
            revisions.append({'old': old_id, 'new': row['configuration_id'], 'applied_roles': len(edits),
                              'exact_scales_resolved': row['exact_scales_resolved']})
        ids = []
        for dependency_id in original['dependencies']:
            if dependency_id not in original_jobs:
                raise ValueError('missing referenced dependency ' + dependency_id)
            dep = deepcopy(original_jobs[dependency_id])
            before = deepcopy(dep['inputs'])
            def rewrite(value):
                if isinstance(value, list):
                    return [rewrite(v) for v in value]
                if not isinstance(value, dict):
                    return value
                value = {k: rewrite(v) for k, v in value.items()}
                for edit in edits:
                    if all(value.get(k) == edit[k] for k in ('role', 'model_key', 'asset_sha256')) and 'applied_scale' in value:
                        if value['applied_scale'] not in (None, edit['applied_scale']):
                            raise ValueError('conflicting dependency role scale')
                        value['applied_scale'] = edit['applied_scale']
                        value['scale_rule'] = 'literal_exact_scene_initialization'
                    # Support descriptors omit asset/model, but carry exact scene object.
                    if (edit['role'] == 'support' and value.get('scene') == row['scene']
                            and value.get('object') in edit['matching_instances'] and 'selected_scale' in value):
                        if value['selected_scale'] not in (None, edit['applied_scale']):
                            raise ValueError('conflicting support scale')
                        value['selected_scale'] = edit['applied_scale']
                return value
            dep['inputs'] = rewrite(dep['inputs'])
            if dep['inputs'] != before:
                identity = {'kind': dep['kind'], 'inputs': dep['inputs']}
                dep['id'] = 'request_' + digest(identity)
                dep['supersedes_request_ids'] = [dependency_id]
                missing = list(dep['unresolved_inputs'])
                if row.get('unresolved_scene_bindings'):
                    missing.append('exact scene object binding among identical-scale instances')
                if (dep['inputs'].get('target') or {}).get('applied_scale') is not None:
                    missing = [x for x in missing if x != 'exact native target default and applied scale']
                dep.update(unresolved_inputs=missing, input_resolution='unresolved' if missing else 'resolved',
                           status='blocked' if missing else 'missing', physical_artifact_id=None if missing else 'physical_' + digest(identity),
                           artifact=None, evidence=[], active_jobs=[],
                           next_action='Verify revised exact input dependency; scene scale evidence does not prove physical readiness')
                dependency_revisions.append({'configuration_id':row['configuration_id'], 'old':dependency_id, 'new':dep['id']})
            key = dep['id']
            if key not in result_jobs:
                dep['consumers'] = []
                dep['required_by_classes'] = []
                result_jobs[key] = dep
            result_jobs[key]['consumers'].append(row['configuration_id'])
            result_jobs[key]['required_by_classes'].append(row['class_id'])
            ids.append(key)
        row['dependencies'] = sorted(set(ids))
        result_rows.append(row)
    for dep in result_jobs.values():
        dep['consumers'] = sorted(set(dep['consumers']))
        dep['required_by_classes'] = sorted(set(dep['required_by_classes']))
    missing = sum(not row['exact_scales_resolved'] for row in result_rows)
    summary = {'configuration_choices':len(result_rows), 'revised_configurations':len(revisions),
               'resolved_configurations':sum(r['exact_scales_resolved'] for r in revisions),
               'exact_scales_missing':missing,'exact_scales_resolved':len(result_rows)-missing,
               'revised_dependency_links':len(dependency_revisions), 'logical_dependencies':len(result_jobs),
               'dependency_statuses':dict(Counter(d['status'] for d in result_jobs.values())),
               'full_dependency_denominator_complete':False,'production_authorized':False}
    return ({'schema':'isg-scale-expansion-revision-v1', 'rows':result_rows, **summary},
            {**{k:v for k,v in jobs.items() if k!='rows'}, 'rows':list(result_jobs.values()),
             'scale_resolution_applied':True,'logical_requirement_count':len(result_jobs),
             'configuration_requirement_links':sum(len(r['dependencies']) for r in result_rows),
             'physical_key_count':sum(d['physical_artifact_id'] is not None for d in result_jobs.values())},
            {'summary':summary,'configuration_revisions':revisions,'dependency_revisions':dependency_revisions})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('expansion','proof','jobs','source-verification','output'):
        p.add_argument('--'+name, required=True, type=Path)
    a=p.parse_args(); paths={k:getattr(a,k.replace('-','_')) for k in ('expansion','proof','jobs','source-verification')}
    docs={k:json.loads(v.read_text()) for k,v in paths.items()}
    if file_ref(a.expansion)['sha256'] != docs['proof']['source']['sha256']:
        raise ValueError('scale expansion source changed')
    if file_ref(a.proof)['sha256'] != docs['source-verification']['source_proof_sha256']:
        raise ValueError('scene evidence changed')
    expansion,jobs,receipt=apply(docs['expansion'],docs['proof'],docs['jobs'],docs['source-verification'])
    a.output.mkdir(parents=True, exist_ok=False)
    receipt.update(observed_at=datetime.now(timezone.utc).isoformat(), sources={k:file_ref(v) for k,v in paths.items()})
    for name,doc in [('scale_expansion.json',expansion),('dependency_jobs.json',jobs),('application_receipt.json',receipt)]:
        (a.output/name).write_text(json.dumps(doc,sort_keys=True,separators=(',',':'))+'\n')
    print(json.dumps(receipt['summary'],sort_keys=True))

if __name__=='__main__':
    main()
