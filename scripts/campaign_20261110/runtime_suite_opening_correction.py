"""Bind a fixed opening goal to its existing strict fraction bucket."""
from pathlib import Path
from .contracts import require
from .runtime_suite import checked
from .runtime_suite_acceptance import reference, template_from


def recorded_opening(root, correction_ref, prior_ref, proposed):
    from .controller_groups import ranges
    audit=checked(root,correction_ref);prior=checked(root,prior_ref)
    require(audit.get('schema')=='isg-current100-articulation-recorded-range-audit-v1' and
            audit.get('owner')=='/root/templates_assets' and audit.get('no_original_mutation') is True and
            audit['run_id']==prior['run_id'], 'unknown opening correction audit')
    matches=[x for x in audit['rows'] if x['class_id']==prior['class_id']]
    require(len(matches)==1, 'opening correction class absent or ambiguous')
    e=matches[0]
    for key,expected in [('original_row',prior_ref),('original_configuration',prior['campaign_config']),
                         ('original_goal_plan',prior['goal_plan']),('catalog_row',prior['catalog_row']),
                         ('frozen_runtime',prior['frozen_runtime'])]:
        require(e[key]==expected,'opening correction input identity differs: '+key)
    require(e.get('opening_mismatch') is True and e.get('direction')=='open' and
            e.get('native_checks_changed') is False,'not a proven fixed-opening mismatch')
    require(proposed['campaign_config']==prior['campaign_config'] and
            proposed.get('catalog_row')==prior.get('catalog_row') and
            proposed.get('effective_limits')==prior.get('effective_limits'), 'opening correction changes configuration or limits')
    checked(root,prior['catalog_row'])
    template=template_from(checked(root,prior['campaign_config']))
    require(template.task in ('configuration_matching-articulation_open_small_scale',
                             'configuration_matching-articulation_open_large_scale') and
            template.split=='eval_standard' and not prior['camera_exempt'], 'correction is opening standard eval only')
    old=checked(root,prior['goal_plan']);new=checked(root,proposed['goal_plan'])
    require(old['schema']==new['schema']=='isg-joint-goal-plan-v1','not an exact joint goal plan')
    targets=[o for o in template.objects if o.role=='target']
    require(len(targets)==1 and targets[0].joint and old['target']==targets[0].instance and
            old['joint']==targets[0].joint,'opening target/joint differs')
    sources=e['source_evidence']
    for name in ('tro','summary','sidecar','job_env'):
        ref=sources[name];p=Path(ref['path']).resolve()
        require(ref==old['anchor_sources'][name] and p.is_relative_to(Path(root).resolve()) and
                reference(p)['sha256']==ref['sha256'],'opening anchor bytes changed')
    metadata=checked(root,e['native_template_metadata']);tro=checked(root,sources['tro']);summary=checked(root,sources['summary'])
    require(Path(e['native_template_metadata']['path']).resolve().parent==Path(sources['tro']['path']).resolve().parent.parent and
            metadata['template_id']==old['template_id'] and summary['direction']=='open' and
            summary['joint']==old['joint'],'opening metadata/source scope differs')
    runtime=tro['ispatialgym']['campaign']['runtime'];target=old['target'];joint=old['joint']
    names=runtime['joint_names'][target];values=runtime['planned_goal_joints'][target]
    require(len(names)==len(values) and len(set(names))==len(names) and
            dict(zip(names,values))==old['goal_joints'],'fixed recorded joint vector changed')
    limits=metadata['template_facts']['limits'];require(len(limits)==2 and limits[0]<limits[1],'invalid joint limits')
    fraction=(old['goal_joints'][joint]-limits[0])/(limits[1]-limits[0])
    candidates=[(list(bounds),i)for i,bounds in enumerate(ranges(template))if bounds[0]<=fraction<=bounds[1]]
    require(len(candidates)==1,'opening goal has no unique strict bucket')
    bounds,bucket=candidates[0];historical=summary['campaign_goal_group']
    require(historical['bucket_strict'] is True and historical['fraction_range']==bounds and
            historical['requested_goal']['joints']==old['goal_joints'] and
            historical['actual_open_fraction']==fraction,'historical fixed-goal bucket disagrees')
    require(e['limits']==limits and e['goal_fraction']==fraction and e['goal_joint']==old['goal_joints'][joint] and
            e['historical_group']==historical and e['authored_range']==old['fraction_range'] and
            e['expected_opening_range']==bounds and not old['fraction_range'][0]<=fraction<=old['fraction_range'][1],
            'audit does not prove incompatible original opening range')
    require(new['fraction_range']==bounds and proposed['selected_bucket']==bucket,'opening correction must use exact existing bucket')
    ignored={'group_id','created_at','fraction_range'}
    require({k:v for k,v in old.items()if k not in ignored}=={k:v for k,v in new.items()if k not in ignored},
            'opening correction changes physical goal, timing, source or checks')
    return {'kind':'recorded_opening_range','audit':correction_ref,'fraction':fraction,
            'strict_range':bounds,'bucket':bucket,'changed_fields':['goal_plan.fraction_range','selected_bucket']}
