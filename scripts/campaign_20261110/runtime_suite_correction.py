"""Verify an explicitly diagnosed authoring correction, never a goal relaxation."""
from dataclasses import asdict
from pathlib import Path
from .contracts import digest, require
from .runtime_suite import checked
from .runtime_suite_acceptance import reference, template_from


def recorded_range(root, correction_ref, prior_ref, proposed):
    """Only the recorded fixed selection's strict range may be corrected."""
    from .runtime_suite_author import recorded_disembedding_evidence
    audit = checked(root, correction_ref)
    require(audit.get('schema') == 'isg-current100-disembedding-recorded-range-audit-v1'
            and audit.get('owner') == '/root/templates_assets'
            and audit.get('native_sampler_changed') is False
            and audit.get('original_rows_unchanged') is True,
            'unknown or physically changed input-correction audit')
    prior = checked(root, prior_ref)
    require(proposed.get('catalog_row') == prior.get('catalog_row')
            and proposed.get('effective_limits') == prior.get('effective_limits'),
            'range correction changes source catalog or execution limits')
    matches = [r for r in audit['rows'] if r['class_id'] == prior['class_id']]
    require(len(matches) == 1, 'correction audit class absent or ambiguous')
    evidence = matches[0]
    require(audit['run_id'] == prior['run_id'] and evidence['original_row'] == prior_ref
            and evidence['original_configuration'] == prior['campaign_config']
            and evidence['original_goal_plan'] == prior['goal_plan']
            and evidence['catalog_row'] == prior['catalog_row']
            and evidence['frozen_runtime'] == prior['frozen_runtime'],
            'correction audit belongs to another original input')
    require(evidence.get('status') == 'input_authoring_mismatch'
            and evidence.get('pool_geometry_goal_change') is False
            and evidence.get('bucket_threshold_change') is False
            and evidence.get('native_sampler_change') is False,
            'correction is not a range-only authoring mismatch')
    checked(root, prior['catalog_row'])
    config = checked(root, prior['campaign_config']); template = template_from(config)
    require(template.task == 'disembedding' and template.split == 'eval_standard'
            and not prior['camera_exempt'], 'range correction is standard disembedding only')
    old = checked(root, prior['goal_plan']); new = checked(root, proposed['goal_plan'])
    require(old['schema'] == new['schema'] == 'isg-tray-goal-plan-v1',
            'correction requires an unchanged native tray plan')
    sources = evidence['source_evidence']
    for name in ('tro', 'summary', 'sidecar', 'job_env'):
        ref = sources[name]
        require(ref == old['anchor_sources'][name], 'correction source differs from original anchor')
        path = Path(ref['path']).resolve()
        require(path.is_relative_to(Path(root).resolve()) and reference(path)['sha256'] == ref['sha256'],
                'correction source bytes changed')
    metadata_ref = evidence['native_template_metadata']
    metadata = checked(root, metadata_ref)
    native_directory = Path(metadata_ref['path']).resolve().parent
    require(Path(sources['tro']['path']).resolve().parent.parent == native_directory
            and metadata['template_id'] == old['template_id'],
            'correction coverage metadata is not the exact source template')
    anchor = {'sources': sources, 'tro': checked(root, sources['tro']),
              'summary': checked(root, sources['summary']), 'template_dir': str(native_directory),
              'template_meta': metadata}
    measured = recorded_disembedding_evidence(template, anchor)
    profile = metadata['template_facts']['campaign_coverage_profile']
    active = [asdict(o) for o in template.objects if o.role in ('target', 'distractor')]
    require(evidence['native_coverage_profile'] == profile
            and evidence['native_coverage_profile_sha256'] == digest(profile)
            and digest(evidence['active_objects']) == digest(active)
            and evidence['active_objects_sha256'] == measured['active_objects_sha256']
            and evidence['measured_coverage'] == measured['coverage']
            and evidence['measured_bucket'] == measured['bucket'],
            'correction audit disagrees with independently measured active geometry')
    pool = config['template']['scene_setup']['binding_pool']
    require(evidence['binding_pool'] == pool and evidence['binding_pool_manifest'] ==
            {'path': pool['manifest'], 'sha256': pool['manifest_sha256']},
            'correction changes the fixed binding pool')
    checked(root, evidence['binding_pool_manifest'])
    bounds = measured['strict_range']
    require(evidence['expected_strict_range'] == bounds
            and evidence['historical_strict_range'] == bounds
            and evidence['historical_actual_coverage'] == measured['historical_actual_crowdedness']
            and evidence['authored_strict_range'] == old['crowdedness_range']
            and old['crowdedness_range'] != bounds
            and new['crowdedness_range'] == bounds
            and proposed['selected_bucket'] == measured['bucket'],
            'correction must select exactly the measured historical strict range')
    ignored = {'group_id', 'created_at', 'crowdedness_range'}
    require({k:v for k,v in old.items() if k not in ignored} ==
            {k:v for k,v in new.items() if k not in ignored},
            'range correction changes physical goal, timing, source or checks')
    return {'kind': 'recorded_disembedding_range', 'audit': correction_ref,
            'measurement': measured, 'changed_fields': ['goal_plan.crowdedness_range', 'selected_bucket']}
