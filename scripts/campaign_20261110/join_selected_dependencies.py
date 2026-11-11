"""Join executed library proofs to selected logical requests without filling gaps.

The request ID remains the selected logical obligation; the physical identity
records the exact measured library and generator component evidence separately.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy

from scripts.campaign_20261110.contracts import digest


def matches(request, row):
    target = request.get('inputs', {}).get('target') or {}
    job = row['job']
    return (target.get('asset_sha256') == job.get('asset_sha256') and
            target.get('category') == job.get('category') and target.get('model') == job.get('model') and
            target.get('dataset') == job.get('dataset') and target.get('applied_scale') == job.get('scale') and
            set(request['required_by_classes']).issubset(job['campaign_classes']))


def join(manifest, audit, audit_ref, component_ref):
    out = deepcopy(manifest)
    joined, failures = [], []
    for request in out['artifact_requests']:
        kind = request['kind']
        if kind not in ('floating_grasp_library', 'floating_native_motion_validation', 'tabletop_grasp_library'):
            continue
        constraint = 'tabletop' if kind == 'tabletop_grasp_library' else 'floating'
        good = [r for r in audit['verified'] if r['job']['constraint'] == constraint and matches(request, r)]
        bad = [r for r in audit['failed'] if r['job']['constraint'] == constraint and matches(request, r)]
        if good:
            row = good[0]
            # Floating motion validation is arm-specific, even when the grasp
            # proposals themselves can subsequently be tried with either arm.
            if kind == 'floating_native_motion_validation':
                if request['inputs'].get('arm') != row.get('validated_arm'):
                    continue
                task_names = {c.split('-', 1)[0] for c in request['required_by_classes']}
                if not task_names.issubset(row['job'].get('held_tasks', [])):
                    continue
            remaining = [x for x in request['unresolved_inputs'] if x != 'physical generator component fingerprint']
            if remaining:
                continue
            identity = dict(kind=kind, asset_sha256=row['job']['asset_sha256'],
                            native_identity_key=row['native_identity_key'], generator_components=component_ref,
                            arm=request['inputs'].get('arm'))
            request.update(status='verified', input_resolution='resolved', unresolved_inputs=[],
                physical_artifact_id='physical_'+digest(identity), artifact=row['library'],
                evidence=[audit_ref, row['library'], component_ref],
                resolved_inputs={'native_library_identity':row['native_identity'], 'generator_components':component_ref},
                active_jobs=[], next_action='Exact dependency verified; remaining configuration gates remain independent',
                observed_at=audit['observed_at'])
            joined.append(request['id'])
        elif bad:
            row=bad[0]
            request.update(status='failed', artifact=None, evidence=[audit_ref,row['library']],
                active_jobs=[], last_failure=row['reason'],
                next_action='Inspect exact native candidate/validation rejection before a changed bounded dependency build',
                observed_at=audit['observed_at'])
            failures.append(request['id'])
    by_id={r['id']:r for r in out['artifact_requests']}
    for config in out['configurations']:
        required=[by_id[k] for k in config['dependencies']]
        config['verified_dependency_count']=sum(r['status'] in ('verified','not_applicable') for r in required)
        config['remaining_dependency_count']=len(required)-config['verified_dependency_count']
        # Complete dependency preparation still does not materialize or validate
        # an exact configuration, much less authorize production.
        config['status']='dependencies_verified_configuration_unmaterialized' if not config['remaining_dependency_count'] else \
            'dependency_failed' if any(r['status']=='failed' for r in required) else config['status']
    for row in out['per_class']:
        configs=[c for c in out['configurations'] if c['class_id']==row['class_id']]
        ids={k for c in configs for k in c['dependencies']}
        row['dependency_statuses']=dict(Counter(by_id[k]['status'] for k in ids))
        row['fully_served']=sum(c['remaining_dependency_count']==0 for c in configs)
    out['summary']['dependency_statuses']=dict(Counter(r['status'] for r in by_id.values()))
    out['summary']['resolved_physical_artifact_identities']=sum(r['physical_artifact_id'] is not None for r in by_id.values())
    out['summary']['fully_served_configurations']=sum(c['remaining_dependency_count']==0 for c in out['configurations'])
    out['summary']['exact_materialized_configurations']=sum(c.get('configuration_sha256') is not None for c in out['configurations'])
    out['observed_at']=audit['observed_at']
    out['native_library_join']={'audit':audit_ref,'generator_components':component_ref,
                               'verified_request_ids':joined,'failed_request_ids':failures}
    return out
