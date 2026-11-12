"""Fresh exact-runtime acceptance, with a fully audited non-consuming regression lane."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import csv
import json
from pathlib import Path
import time
from .contracts import ObjectRef, Template, digest, file_digest, require, task_quotas, write_json_atomic
from .runtime_suite import authorize, checked, timestamp


def reference(path):
    return {'path':str(Path(path).resolve()), 'sha256':file_digest(path)}


def immutable(path, value):
    path = Path(path)
    if path.exists():
        require(json.loads(path.read_text()) == value, 'existing immutable suite evidence differs')
    else:
        write_json_atomic(path, value)
    return reference(path)


def template_from(config):
    value = dict(config['template'])
    value['objects'] = tuple(ObjectRef(**o) for o in value['objects'])
    return Template(**value)


def allocated_scope(root, row_ref, allocation_ref):
    """Verify dispatch identity even when the worker was canceled before startup."""
    root = Path(root).resolve(); row = checked(root,row_ref); wrapper = checked(root,allocation_ref)
    require(wrapper['row'] == row_ref, 'allocation references another suite row')
    allocation = wrapper['allocation']; job = Path(allocation['job_dir']).resolve()
    require(job.is_relative_to(root/'out') and
            Path(allocation['code']).resolve() == Path(row['frozen_runtime']['path']).resolve() and
            (int(allocation['start']),int(allocation['n'])) == (row['fresh_start'],row['fresh_count']),
            'suite executed allocation differs')
    receipt = checked(root,wrapper['dispatch_receipt'])
    require(receipt['status'] == 'submitted' and all(str(receipt['allocation'][k]) == str(allocation[k])
            for k in ('jobid','job_dir','code','start','n')), 'suite dispatch allocation differs')
    ledger = Path(receipt['allocation']['ledger']).resolve()
    require(ledger.is_relative_to(root/'manifests'), 'suite allocation ledger outside campaign')
    with ledger.open() as stream:
        matches = [r for r in csv.DictReader(stream,delimiter='\t') if r.get('jobid') == str(allocation['jobid'])]
    require(len(matches) == 1 and all(str(matches[0][k]) == str(allocation[k])
            for k in ('jobid','job_dir','code','start','n')), 'suite allocation is absent or ambiguous in its ledger')
    return row, wrapper, checked(root,row['campaign_config']), job


def executed_scope(root, row_ref, allocation_ref):
    from .production_gate import read_environment
    from .training_diversity import _terminal_jobs
    row, wrapper, config, job = allocated_scope(root, row_ref, allocation_ref)
    allocation = wrapper['allocation']
    environment = read_environment(job/'job.env')
    require(environment.get('ISPATIALGYM_RUNTIME_SUITE_ROW_SHA256') == row_ref['sha256'] and
            Path(environment['ISPATIALGYM_RUNTIME_SUITE_ROW']).resolve() == Path(row_ref['path']).resolve(),
            'worker environment belongs to another row')
    authorize(root,environment,row['campaign_config']['path'],config,
              start=int(environment['START']),count=int(environment['N']),kind='source_validation')
    terminal = checked(root,wrapper['terminal_evidence'])
    require(terminal['observed_at'] >= timestamp(row['run_started_at']), 'suite terminal evidence predates run')
    inventory = [j for j in terminal['jobs'] if j['job_id']==str(allocation['jobid']) and j['stage'] in ('native','collection')]
    require(len(inventory)==1,
            'terminal evidence omits the actual collection allocation')
    _terminal_jobs(root,{'jobs':inventory},{**terminal,'jobs':inventory},terminal['observed_at'])
    return row,wrapper,config,job


def scope(root, row_ref, allocation_ref, gate_ref):
    row,wrapper,config,job=executed_scope(root,row_ref,allocation_ref)
    gate = checked(root,gate_ref)
    require(Path(gate['source_job']).resolve() == job, 'suite requires its own newly executed replay gate')
    require(Path(gate['snapshot']).resolve() == Path(row['frozen_runtime']['path']).resolve(),
            'suite replay used another frozen runtime')
    require(gate.get('success') is True and gate.get('replay_rc') == 0, 'suite replay did not pass with clean native exit')
    selected_replay_terminal(root,wrapper,[gate_ref])
    return row, wrapper, config, job, gate


def selected_replay_terminal(root, wrapper, gate_refs):
    """Optional third-member work remains in cost history without blocking two passes."""
    from .training_diversity import _terminal_jobs
    terminal=checked(root,wrapper['terminal_evidence']);selected=[]
    for gate_ref in gate_refs:
        output=Path(gate_ref['path']).resolve().parent;matches=[]
        for record in wrapper.get('records',[]):
            if record['stage']!='physics_replay' or Path(record['job_dir']).resolve()!=output:continue
            owner=checked(root,record['submission_ref'])
            if str(owner.get('replay_job_id'))==str(record['job_id']) and Path(owner['output']).resolve()==output:
                matches.append(record)
        require(len(matches)==1,'selected member lacks exact current replay allocation owner')
        record=matches[0]
        jobs=[j for j in terminal['jobs'] if j['job_id']==str(record['job_id']) and
              j['backend']==record['backend'] and j['stage']=='replay']
        require(len(jobs)==1,'selected member lacks its actual terminal replay proof')
        selected.extend(jobs)
    selected=list({(j['backend'],j['job_id'],j['stage']):j for j in selected}.values())
    _terminal_jobs(root,{'jobs':selected},{**terminal,'jobs':selected},terminal['observed_at'])
    return selected


def audit(root, row_ref, allocation_ref, gate_ref):
    from .acceptance import audit_job
    from .camera_lane import audit_camera
    from .group_lane import claim_group, check_reservations
    from .reservation_policy import active_policy
    from .current_eligibility import pending_instance_blockers
    from .training_diversity import _index
    row, allocation, config, job, gate = scope(root,row_ref,allocation_ref,gate_ref)
    template = template_from(config); since = timestamp(row['run_started_at'])
    policy = active_policy(root); require(policy is not None, 'suite needs active reservation evidence')
    check_reservations(template,policy[1],policy[2])
    if row['camera_exempt']:
        review = audit_camera(job,gate_ref['path'],root=root,started_at=since,template=template)
        require(len(review['members']) == 1, 'camera suite expects one successful exact member')
        validation = {'validation_passed':True,'quota_consumed':0,'camera_exempt':True,'members':review['members']}
    else:
        review = audit_job(job,gate_ref['path'],root=root,started_at=since,group_closed=True,group_id=row['group_id'])
        require(2 <= len(review['members']) <= 3, 'suite needs at least two distinct successful members')
        replay_proofs=member_replays(root,row,allocation,review)
        selected_replay_terminal(root,allocation,list(allocation['member_gates']))
        review=audit_job(job,gate_ref['path'],root=root,started_at=since,group_closed=True,
                         group_id=row['group_id'],member_ids=sorted(replay_proofs))
        validation = claim_group(review,template,policy[1],reservation_hashes=policy[2],root=root,
            started_at=since,state_path=Path(root)/'dataset/allocations.sqlite',validation_only=True,
            suite_context={'row':row_ref,'allocation':allocation_ref,'gate':gate_ref})
        require(validation.get('validation_passed') is True, 'strict regression audit did not pass')
        validation['member_replays']=replay_proofs
    require(review['campaign_id'] == 'campaign_20261110' and review['class_id'] == row['class_id'] and review['validation_key'] == config['validation_key'],
            'suite actual review differs from the exact new configuration')
    require(not pending_instance_blockers(root,[m['instance_id'] for m in review['members']]),
            'suite members have unresolved change-impact decisions')
    require(all(_index(m) is not None and row['fresh_start'] <= _index(m) < row['fresh_start']+row['fresh_count'] for m in review['members']),
            'suite member outside fresh reserved range')
    return row, allocation, template, review, validation


def member_replays(root,row,allocation,review):
    from .acceptance import replay_report_path, report_passed
    from .runtime_suite import timestamp
    from .replay_gate_quality import assess
    members={m['instance_id']:m for m in review['members']}; proven={}; seen=set()
    excluded=review.get('excluded_duplicate_members',[])
    require(all(m.get('reason')=='duplicate_start_grasp_configuration' and
                m['instance_id'] not in members for m in excluded),
            'suite excluded member is not an audited duplicate')
    references=allocation.get('member_gates',[])
    require(isinstance(references,list) and len(references)>=2,'awaiting_member_replay: suite needs two actual member gates')
    for gate_ref in references:
        gate=checked(root,gate_ref)
        quality=assess(gate,gate_ref['path'])
        require(quality['passed'] is True, 'suite native replay quality failed: '+str(quality['reasons']))
        require(gate.get('success') is True and gate.get('replay_mode')=='physics_action_open_loop' and
                Path(gate['source_job']).resolve()==Path(review['source_job']).resolve() and
                Path(gate['snapshot']).resolve()==Path(row['frozen_runtime']['path']).resolve(),
                'suite member replay source/runtime differs or failed')
        matches=[m for m in list(members.values())+excluded if Path(m['tro']).resolve()==Path(gate['instance']).resolve()]
        require(len(matches)==1,'suite member replay does not match a successful distinct demo')
        member=matches[0];iid=member['instance_id']
        require(iid not in seen and gate['tro_sha256']==member['tro_sha256'] and
                gate['demo_sha256']==member['demo_sha256'] and Path(gate['demo']).resolve()==Path(member['demo']).resolve(),
                'suite member replay TRO/demo identity differs or is duplicated')
        path=replay_report_path(gate,gate_ref['path'],iid,root=Path(root).resolve(),started_at=timestamp(row['run_started_at']))
        report=json.loads(path.read_text());report_passed(report,'replay')
        require(Path(report['provenance']['isg_root']).resolve()==Path(row['frozen_runtime']['path']).resolve() and
                timestamp(report['started_utc'])>=timestamp(row['run_started_at']), 'member replay used old runtime or evidence')
        seen.add(iid)
        # The full audit already excluded this exact successful duplicate start.
        # Verify its replay identity above, retain its cost, but never count it
        # as an independent sibling or let it block two distinct valid members.
        if iid in members:
            proven[iid]={'gate':gate_ref,'report':reference(path),'tro_sha256':member['tro_sha256'],'demo_sha256':member['demo_sha256']}
    require(len(proven)>=2,'awaiting_member_replay: fewer than two distinct native replay successes')
    return proven


def verify_claim_context(root,review,context):
    row,allocation,config,job,gate=scope(root,context['row'],context['allocation'],context['gate'])
    require(not row['camera_exempt'] and Path(review['source_job']).resolve()==job and
            review['validation_key']==config['validation_key'] and review['group_id']==row['group_id'],
            'suite claim review differs from exact executed row')
    proofs=member_replays(root,row,allocation,review)
    selected_replay_terminal(root,allocation,list(allocation['member_gates']))
    require(review.get('selected_member_ids')==sorted(proofs), 'suite claim includes a member without its actual successful replay')
    return proofs


def quota_snapshot(root, template, members):
    from .export_release import snapshot_allocations
    from .current_eligibility import assess
    rows,groups,inputs = snapshot_allocations(root)
    outcomes,_ = assess(root,rows,groups)
    eligible = [r for r in rows if outcomes[r['instance_id']]['quota_eligible']]
    same = [r for r in eligible if r['split']=='eval_standard' and r['task']==template.task and
            r.get('class_id',template.task)==template.class_id]
    counts = Counter(str(r['bucket']) for r in same)
    quota = next(q for q in task_quotas() if q.task==template.task)
    limits = dict(zip(('3','5','10') if template.task.startswith('camera_') else
                     (str(i) for i in range(len(quota.standard_buckets or (10,)))), quota.standard_buckets or (10,)))
    available = len(same)<10 and any(counts[str(m['bucket'])] < limits[str(m['bucket'])] for m in members)
    return {'observed_at':time.time(),'inputs':inputs,'eligible_before':len(same),'bucket_counts':dict(counts),
            'bucket_limits':limits,'eligible':available,'canonical_members':same}, eligible, groups


def _claim_camera(root, template, review, since, policy):
    from .camera_lane import claim_camera, room_caps
    from .camera_classes import claim_camera_classes
    from .runtime_suite import active
    root = Path(root); plan=json.loads((root/'manifests/train_room_plan_20261110.json').read_text())
    quota=next(q for q in task_quotas() if q.task==template.task)
    caps=room_caps(plan,total=quota.train_instances,task=template.task)
    if template.task=='camera_relocalization-standard':
        return claim_camera(review,root/'dataset/camera_quota.json',root=root,started_at=since,training_room_caps=caps)
    manifest=active(root)[2]
    classes=[r['class_id'] for r in manifest['classes'] if r['class_id'].startswith(template.task+'-')]
    return claim_camera_classes(review,root/'dataset/camera_obstructed_quota.json',root=root,started_at=since,
        standard_classes=classes,training_room_caps=caps,reservations=policy[1],reservation_hashes=policy[2])


def reconcile(root, row_ref, allocation_ref, gate_ref, *, claim=True):
    from .group_lane import claim_group
    from .reservation_policy import active_policy
    from .runtime_suite import active
    base=active(root)[4]
    row_identity=checked(root,row_ref)
    target=base/'acceptance'/(row_identity['class_id']+'.json')
    if target.exists():
        prior=checked(root,reference(target))
        if prior.get('status')=='accepted' and prior.get('row')==row_ref:
            # A later optional replay updates cost history, not the immutable
            # already-committed group review. Re-audit its exact selected pair.
            return verify_result(root,reference(target))
    row,allocation,template,review,validation = audit(root,row_ref,allocation_ref,gate_ref)
    work=base/'evidence'/row['class_id']/digest({'row':row_ref,'allocation':allocation_ref,'gate':gate_ref})
    review_ref=immutable(work/'review.json',review); validation_ref=immutable(work/'regression.json',validation)
    quota,_,_=quota_snapshot(root,template,validation['members'])
    allocation_result={'accepted':{}}
    if quota['eligible'] and claim:
        policy=active_policy(root)
        allocation_result=(_claim_camera(root,template,review,timestamp(row['run_started_at']),policy) if row['camera_exempt'] else
            claim_group(review,template,policy[1],reservation_hashes=policy[2],root=root,
                started_at=timestamp(row['run_started_at']),state_path=Path(root)/'dataset/allocations.sqlite',
                suite_context={'row':row_ref,'allocation':allocation_ref,'gate':gate_ref}))
    # Read actual canonical assignments after the ordinary owner transaction.
    after,rows,groups=quota_snapshot(root,template,validation['members'])
    ids={m['instance_id'] for m in review['members']}
    accepted={r['instance_id']:r['split'] for r in rows if r['instance_id'] in ids}
    complete=(bool(accepted) if row['camera_exempt'] else
              list(accepted.values()).count('eval_standard')==1 and list(accepted.values()).count('train')>=1)
    quota_ineligible=not quota['eligible'] and not accepted
    status='accepted' if complete else ('regression_validation_passed' if quota_ineligible else 'validated_pending_acceptance')
    quota.update(accepted=accepted,consumed=len(accepted),quota_ineligible=quota_ineligible)
    quota_ref=immutable(work/('quota_'+str(time.time_ns())+'.json'),quota)
    gate_instance=json.loads(Path(gate_ref['path']).read_text())['instance']
    members=[]
    for member in validation['members']:
        tro=member.get('tro') or member['artifacts']['tro']['path']
        member_proof=validation.get('member_replays',{}).get(member['instance_id'])
        members.append({**member,'demo_passed':True,'physics_replay_passed':True,
            'physics_replay_applicability':'own_member_replay',
            'physics_gate':member_proof['gate'] if member_proof else gate_ref,
            'member_replay':member_proof})
    result={'schema':'isg-current-runtime-class-acceptance-v1','campaign_id':'campaign_20261110',
        'run_id':row['run_id'],'run_started_at':row['run_started_at'],'class_id':row['class_id'],
        'observed_at':time.time(),'status':status,'completed':complete or quota_ineligible,
        'physical_validation_complete':True,
        'validation_passed':complete or quota_ineligible,'physical_validation_passed':True,
        'frozen_runtime':row['frozen_runtime'],'campaign_config':row['campaign_config'],
        'group_id':row['group_id'],'fresh_start':row['fresh_start'],'fresh_count':row['fresh_count'],
        'camera_exempt':row['camera_exempt'],'row':row_ref,'allocation':allocation_ref,'gate':gate_ref,
        'strict_review':review_ref,'regression_audit':validation_ref,'quota_evidence':quota_ref,
        'members':members,'distinct_successful_members':None if row['camera_exempt'] else len(members),
        'quota':quota,'claim_result':allocation_result,'production_authorization':False}
    if target.exists():
        prior=json.loads(target.read_text());immutable(work/('prior_'+digest(prior)+'.json'),prior)
    immutable(work/('result_'+digest(result)+'.json'),result)
    write_json_atomic(target,result)
    return result


def verify_result(root, result_ref):
    """Repeat physical audit; verify stored assignment against committed history."""
    result=checked(root,result_ref)
    require(result.get('schema')=='isg-current-runtime-class-acceptance-v1' and result['validation_passed'] is True,
            'current-runtime class lacks a completed strict result')
    row,allocation,template,review,validation=audit(root,result['row'],result['allocation'],result['gate'])
    require(checked(root,result['strict_review'])==review and checked(root,result['regression_audit'])==validation,
            'suite physical evidence changed after verification')
    for key in ('run_id','run_started_at','class_id','frozen_runtime','campaign_config','group_id','fresh_start','fresh_count','camera_exempt'):
        require(result[key]==row[key], 'suite result identity differs')
    quota=checked(root,result['quota_evidence']);require(quota==result['quota'],'suite quota evidence changed')
    if result['status']=='regression_validation_passed':
        require(quota['quota_ineligible'] is True and quota['eligible'] is False and
                quota['consumed']==0 and quota['accepted']=={},'regression result consumed quota')
        current,_,_=quota_snapshot(root,template,validation['members'])
        require(current['eligible'] is False, 'quota became eligible; reconcile this validated group normally')
    else:
        require(result['status']=='accepted','suite acceptance is pending')
        _,rows,groups=quota_snapshot(root,template,validation['members'])
        actual={r['instance_id']:r['split'] for r in rows if r['instance_id'] in quota['accepted']}
        require(actual==quota['accepted'] and (bool(actual) if row['camera_exempt'] else
                list(actual.values()).count('eval_standard')==1 and list(actual.values()).count('train')>=1),
                'suite accepted eval/sibling is absent from canonical history')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--row',required=True)
    p.add_argument('--allocation',required=True);p.add_argument('--gate',required=True)
    a=p.parse_args();result=reconcile(a.root,reference(a.row),reference(a.allocation),reference(a.gate))
    print(json.dumps({k:result[k] for k in ('class_id','status','validation_passed','quota')}))


if __name__=='__main__':main()
