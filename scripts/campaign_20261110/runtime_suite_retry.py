"""Explicit, bounded retry registration. Preparing an attempt never dispatches it."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import stat
import tempfile
import time
from .contracts import canonical_json, digest, file_digest, require
from .runtime_suite import active, checked, registered_rows, timestamp
from .runtime_suite_acceptance import executed_scope, immutable, reference
from .storage import atomic_directory_lock


def _replace_pointer(path, value):
    """Retain the existing pointer's access mode across an atomic update.

    NamedTemporaryFile starts with mode0600 regardless of umask. Apply the
    existing regular file's mode before publication, without widening it or
    changing the canonical JSON representation.
    """
    path = Path(path)
    previous = path.lstat()
    require(stat.S_ISREG(previous.st_mode), 'suite pointer must be a regular file')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix=path.name+'.',
                                         delete=False) as stream:
            temporary = Path(stream.name)
            if os.fstat(stream.fileno()).st_gid != previous.st_gid:
                os.fchown(stream.fileno(), -1, previous.st_gid)
            os.fchmod(stream.fileno(), stat.S_IMODE(previous.st_mode))
            stream.write(canonical_json(value)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _share_registered_controls(root, base, entry, registry_ref):
    """Publish only the exact control records read by this registration's gate.

    This is an explicit list of checked references, not a recursive filesystem
    permission walk. Native assets, external paths and unrelated attempts are
    never permission targets. Existing directories must already be traversable.
    """
    prepared = checked(root, entry['prepared'])
    row = checked(root, entry['row'])
    diagnosis = checked(root, prepared['diagnosis'])
    references = [registry_ref, entry['row'], entry['prepared'], entry['approval'],
                  prepared['diagnosis'], row['campaign_config'], row.get('goal_plan'),
                  row.get('catalog_row'), prepared.get('input_correction'),
                  *diagnosis['evidence']]
    group_id = row.get('group_id')
    exact_group_goal = (Path(root)/'groups'/group_id/'goal_plan.json'
                        if isinstance(group_id, str) and Path(group_id).name == group_id
                        else None)
    planned = {}
    for ref in references:
        if ref is None:
            continue
        path = Path(ref['path'])
        # The one external control exception is this row's canonical group goal.
        group_goal = (ref == row.get('goal_plan') and path == exact_group_goal and
                      path.resolve() == exact_group_goal)
        if (not path.resolve().is_relative_to(base) and not group_goal) or path.suffix != '.json':
            continue
        info = path.lstat()
        require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid(),
                'registered control must be an owned regular file')
        require(file_digest(path) == ref['sha256'], 'registered control bytes changed')
        for parent in path.parents:
            boundary = Path(root) if group_goal else base
            if parent == boundary or parent.is_relative_to(boundary):
                require(parent.stat().st_mode & stat.S_IXGRP,
                        'registered control directory is not group traversable')
        planned[str(path)] = (path, ref, stat.S_IMODE(info.st_mode))
    changes = []
    for path, ref, mode in planned.values():
        if not mode & stat.S_IRGRP:
            os.chmod(path, mode | stat.S_IRGRP)
        require(file_digest(path) == ref['sha256'], 'registered control changed during publication')
        changes.append({'reference': ref, 'mode_before': mode,
                        'mode_after': stat.S_IMODE(path.stat().st_mode)})
    return changes


def _reservation(root, proof, row, initial):
    """The shared monotonic allocator, not an arbitrary JSON interval, owns indices."""
    require(proof.get('token') and proof['start']==row['fresh_start'] and
            proof['reserved_count']==row['fresh_count'], 'retry reservation range differs')
    state=json.loads((Path(root)/'manifests/family_index_reservations.json').read_text())
    owned=state['reservations'][proof['token']]
    require((owned['start'],owned['n'])==(proof['start'],proof['reserved_count']),
            'retry indices are not owned by the shared allocator')
    lo,hi=proof['start'],proof['start']+proof['reserved_count']
    original=initial['proof'];a=original['start'];b=a+original['reserved_count']
    require(hi<=a or lo>=b, 'retry overlaps original suite reservation')
    for token,other in state['reservations'].items():
        if token!=proof['token']:
            require(hi<=other['start'] or lo>=other['start']+other['n'],
                    'retry overlaps another reserved interval')


def _row(root, row_ref, prior_ref, number, reservation, pointer, base, correction_ref=None):
    row=checked(root,row_ref);prior=checked(root,prior_ref)
    require(type(number) is int and 1<=number<=2 and row.get('attempt_number')==number,
            'retry attempt number exceeds bounded budget')
    require(Path(row_ref['path']).resolve()==base/'attempts'/row['class_id']/('attempt_'+str(number))/'row.json',
            'retry row path differs')
    for key in ('run_id','run_started_at','release','frozen_runtime','campaign_config','class_id','fresh_count','camera_exempt'):
        require(row[key]==prior[key], 'retry changes immutable '+key)
    require(row['release']==pointer['release'] and row.get('previous_row')==prior_ref,
            'retry belongs to another release or prior attempt')
    require(row['fresh_start']!=prior['fresh_start'], 'retry reuses original indices')
    require(row.get('input_correction')==correction_ref, 'retry correction binding differs')
    if correction_ref:
        schema=checked(root,correction_ref).get('schema')
        if schema=='isg-current100-disembedding-recorded-range-audit-v1':
            from .runtime_suite_correction import recorded_range
            correction_proof=recorded_range(root,correction_ref,prior_ref,row)
        elif schema=='isg-current100-articulation-recorded-range-audit-v1':
            from .runtime_suite_opening_correction import recorded_opening
            correction_proof=recorded_opening(root,correction_ref,prior_ref,row)
        else:raise ValueError('unknown exact input correction type')
    else:
        require(row.get('selected_bucket')==prior.get('selected_bucket'), 'retry changes selected strict bucket')
    _reservation(root,reservation,row,checked(root,pointer['index_reservation']))
    if row['camera_exempt']:
        require(row['goal_plan'] is None and row['group_id'] is None and row['fresh_count']==1,
                'retry camera exception changed')
    else:
        require(row['fresh_count']==3 and row['group_id'] and row['group_id']!=prior['group_id'],
                'retry requires a fresh distinct group identity')
        old=checked(root,prior['goal_plan']);new=checked(root,row['goal_plan'])
        require(new['group_id']==row['group_id'], 'retry goal group binding differs')
        # New indices/group and a measured time allowance may change; target,
        # geometry, source proposals, strict bucket, and every QA option may not.
        ignored={'group_id','created_at','budget_seconds'}
        if correction_ref:ignored.update(k.split('.',1)[1]for k in correction_proof['changed_fields']if k.startswith('goal_plan.'))
        require({k:v for k,v in old.items() if k not in ignored}==
                {k:v for k,v in new.items() if k not in ignored}, 'retry changes physical goal or checks')
        if 'budget_seconds' in new:
            require(type(new['budget_seconds']) in (int,float) and 0<new['budget_seconds']<=2700,
                    'retry sampling time exceeds bounded allowance')
    return row


def failure(root, previous_row, allocation_ref, failed_gates, *, correction_ref=None):
    """Require a terminal actual attempt; absent evidence is not a failure."""
    from .runtime_suite_observer import source_stage
    from .training_diversity import _terminal_jobs
    if correction_ref:
        from .runtime_suite_acceptance import allocated_scope
        row,allocation,config,job=allocated_scope(root,previous_row,allocation_ref)
    else:
        row,allocation,config,job=executed_scope(root,previous_row,allocation_ref)
    terminal=checked(root,allocation['terminal_evidence'])
    expected={(str(record['job_id']),record['backend']) for record in allocation.get('records',[])}
    require(expected<={(j['job_id'],j['backend']) for j in terminal['jobs']},
            'retry terminal evidence omits an owned stage job')
    require(any(j['job_id']==str(allocation['allocation']['jobid']) and j['stage'] in ('native','collection')
                for j in terminal['jobs']), 'retry terminal evidence omits source allocation')
    # Acceptance may select two successes early. A new attempt instead waits
    # for every owned prior job, including optional member replays, to finish.
    _terminal_jobs(root,{'jobs':terminal['jobs']},terminal,time.time())
    receipt=checked(root,allocation['dispatch_receipt'])
    if receipt.get('token'):
        current=Path(root)/'dispatch/receipts'/(receipt['token']+'.json')
        require(current.exists(), 'retry current dispatch owner is missing')
        current_receipt=json.loads(current.read_text())
        require(not current_receipt.get('migration') and current_receipt.get('allocation')==receipt['allocation'],
                'retry must reconcile existing migration ownership first')
    if correction_ref:
        # A positively proven invalid input may be canceled before execution.
        # This is never evidence of a physical failure, and does not authorize
        # an ordinary no-start retry or bypass terminal migration ownership.
        return {'failure_kind':'input_authoring_mismatch','input_correction':correction_ref,
                'all_prior_jobs_terminal':True,'physical_failure_inferred':False}
    stage=source_stage(root,row,allocation)
    require(stage['source_terminal_verified'], 'retry source is not terminal')
    require(stage['status']!='native_not_started_terminal', 'no-start source needs routing reconciliation, not a fresh attempt')
    if stage['status'] in ('native_failed','demo_failed','insufficient_successful_demos'):
        return {'source_stage':stage,'failure_kind':stage['status']}
    require(stage['status']=='awaiting_own_replay' and failed_gates, 'retry lacks an actual terminal failure')
    gates=[]
    for ref in failed_gates:
        gate=checked(root,ref)
        require(gate.get('success') is False and gate.get('replay_rc') is not None and
                Path(gate['source_job']).resolve()==job and
                Path(gate['snapshot']).resolve()==Path(row['frozen_runtime']['path']).resolve(),
                'retry failed replay source/runtime differs or is not terminal')
        tro=Path(gate['instance']).resolve();demo=Path(gate['demo']).resolve()
        require(tro.is_relative_to(job/'inst') and demo.is_relative_to(job/'demo') and
                reference(tro)['sha256']==gate['tro_sha256'] and reference(demo)['sha256']==gate['demo_sha256'],
                'retry failed replay member bytes differ')
        require(timestamp(gate.get('validated_at',gate.get('observed_at',0)))>=timestamp(row['run_started_at']),
                'retry failed replay predates suite')
        gates.append(ref)
    # A failed third member does not invalidate two successful distinct replays.
    successes=[checked(root,g) for g in allocation.get('member_gates',[])]
    require(len({g.get('instance') for g in successes if g.get('success') is True}) < (1 if row['camera_exempt'] else 2),
            'attempt already has sufficient successful member replays')
    return {'source_stage':stage,'failure_kind':'member_replay_failed','failed_gates':gates}


def _diagnosis(root, ref, previous_row, allocation_ref, failed_gates):
    value=checked(root,ref)
    require(value.get('schema')=='isg-runtime-suite-retry-diagnosis-v1' and
            value.get('owner')=='/root' and value.get('previous_row')==previous_row and
            value.get('allocation')==allocation_ref and value.get('failed_gates',[])==failed_gates and
            isinstance(value.get('changed_reason'),str) and len(value['changed_reason'].strip())>=20 and
            isinstance(value.get('next_action'),str) and value['next_action'].strip(),
            'retry needs exact owner diagnosis and concrete changed next action')
    require(value.get('physical_runtime_change') is False and value.get('configuration_change') is False,
            'physical changes require a new current-runtime release')
    require(isinstance(value.get('evidence'),list) and value['evidence'], 'retry diagnosis lacks hashed evidence')
    for artifact in value['evidence']:
        path=Path(artifact['path']).resolve()
        require(path.is_relative_to(Path(root).resolve()) and reference(path)['sha256']==artifact['sha256'],
                'retry diagnosis artifact bytes changed or outside campaign')
    return value


def prepare(root, proposal_ref):
    """Validate existing proposed bytes and save a reviewable prepared authorization."""
    pointer,release,manifest,initial,base=active(root);proposal=checked(root,proposal_ref)
    require(proposal.get('schema')=='isg-runtime-suite-retry-proposal-v1', 'unknown retry proposal')
    row=checked(root,proposal['row']);rows=registered_rows(root,pointer,base,row['class_id'])
    require(proposal['previous_row']==rows[-1], 'retry does not follow current attempt')
    number=len(rows);budget=proposal['max_attempts']
    require(type(budget) is int and 2<=budget<=3 and number<budget, 'retry attempt budget exhausted')
    if number>1:
        previous=checked(root,pointer['attempt_registry'])['attempts'][row['class_id']][-1]
        require(budget==previous['max_attempts'], 'retry cannot expand registered attempt budget')
    correction=proposal.get('input_correction')
    _row(root,proposal['row'],rows[-1],number,proposal['reservation'],pointer,base,correction)
    diagnosis=_diagnosis(root,proposal['diagnosis'],rows[-1],proposal['allocation'],proposal.get('failed_gates',[]))
    require(diagnosis.get('input_correction')==correction, 'diagnosis correction binding differs')
    measured=failure(root,rows[-1],proposal['allocation'],proposal.get('failed_gates',[]),correction_ref=correction)
    result_path=base/'acceptance'/(row['class_id']+'.json')
    if result_path.exists():
        result=json.loads(result_path.read_text())
        require(result.get('validation_passed') is not True, 'validated class does not need retry')
    prepared={'schema':'isg-runtime-suite-prepared-retry-v1','release':pointer['release'],
              'proposal':proposal_ref,'row':proposal['row'],'previous_row':rows[-1],
              'attempt_number':number,'max_attempts':budget,'reservation':proposal['reservation'],
              'diagnosis':proposal['diagnosis'],'allocation':proposal['allocation'],
              'failed_gates':proposal.get('failed_gates',[]),'failure':measured,
              'dispatch_authorized':False,'production_authorization':False}
    if correction:prepared['input_correction']=correction
    # Fresh terminal observations may need a new review revision while the
    # proposed row remains unchanged. Preserve every prepared revision.
    return immutable(Path(proposal['row']['path']).parent/'prepared_authorizations'/(digest(prepared)+'.json'),prepared)


def verify_registered(root, entry, prior_ref, number, pointer, base):
    prepared=checked(root,entry['prepared']);approval=checked(root,entry['approval'])
    require(prepared.get('schema')=='isg-runtime-suite-prepared-retry-v1' and
            prepared['release']==pointer['release'] and prepared['row']==entry['row'] and
            prepared['previous_row']==prior_ref and prepared['attempt_number']==number and
            prepared['max_attempts']==entry['max_attempts'] and number<entry['max_attempts']<=3,
            'registered retry chain/budget differs')
    require(approval.get('schema')=='isg-runtime-suite-retry-approval-v1' and approval.get('owner')=='/root' and
            approval.get('approved') is True and approval.get('prepared')==entry['prepared'] and
            approval.get('diagnosis')==prepared['diagnosis'] and approval.get('automatic_retry') is False,
            'retry lacks explicit exact approval')
    diagnosis=_diagnosis(root,prepared['diagnosis'],prior_ref,prepared['allocation'],prepared['failed_gates'])
    correction=prepared.get('input_correction')
    require(diagnosis.get('input_correction')==correction and
            approval.get('input_correction')==correction, 'approval/diagnosis correction binding differs')
    _row(root,entry['row'],prior_ref,number,prepared['reservation'],pointer,base,correction)
    return prepared


def register(root, prepared_ref, approval_ref):
    """Explicit owner action only; this API never authors or dispatches a job."""
    root=Path(root).resolve()
    with atomic_directory_lock(root/'runtime_validation/retry_registration.lockdir'):
        pointer,release,manifest,initial,base=active(root)
        prepared=checked(root,prepared_ref);row=checked(root,prepared['row'])
        rows=registered_rows(root,pointer,base,row['class_id'])
        entry={'row':prepared['row'],'prepared':prepared_ref,'approval':approval_ref,
               'max_attempts':prepared['max_attempts']}
        verify_registered(root,entry,rows[-1],len(rows),pointer,base)
        require(prepare(root,prepared['proposal'])==prepared_ref,'retry evidence changed after preparation')
        registry=(copy.deepcopy(checked(root,pointer['attempt_registry'])) if pointer.get('attempt_registry') else
                  {'schema':'isg-runtime-suite-attempt-registry-v1','release':pointer['release'],'attempts':{}})
        registry['attempts'].setdefault(row['class_id'],[]).append(entry)
        registry_ref=immutable(base/'attempt_registries'/(digest(registry)+'.json'),registry)
        immutable(base/'pointer_history'/(digest(pointer)+'.json'),pointer)
        access=_share_registered_controls(root,base,entry,registry_ref)
        _replace_pointer(root/'runtime_validation/current.json',{**pointer,'attempt_registry':registry_ref})
        return {'registered':entry,'registry':registry_ref,'dispatch_performed':False,
                'control_access':access}
