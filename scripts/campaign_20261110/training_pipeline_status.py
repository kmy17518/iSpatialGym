"""Read-only projection of bounded pipeline attempts; never author or submit work."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import re
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
from scripts.campaign_20261110.contracts import file_digest, require, write_json_atomic, task_quotas, digest


def read(path): return json.loads(Path(path).read_text())
def ref(path): return {'path': str(path), 'sha256': file_digest(path)}
def timestamp(path): return datetime.fromtimestamp(Path(path).stat().st_mtime, timezone.utc).isoformat()


def sequence_stage_references(root, sequence_path, stage):
    """Check immutable stage identities, including explicit append-journal repair.

    This checks evidence identities for reporting, not physics or release.
    """
    root=Path(root).resolve()
    def checked(reference):
        path=Path(reference['path']).resolve()
        require(path.is_relative_to(root) and file_digest(path)==reference['sha256'],
                'sampler stage evidence changed')
        return reference
    result={'native_gate':checked(stage['native_gate']),
            'template_export':checked(stage['isolation']['template_export'])}
    original=stage['isolation']['pool_execution']
    repair=sequence_path.parent/'immutable_stage_evidence_r1/audit.json'
    if repair.exists():
        audit=read(repair)
        require(audit['original_sequence_result']==ref(sequence_path) and
                audit['status']=='all_recorded_stage_hashes_reproduced',
                'sampler stage reconciliation identity differs')
        matches=[r for r in audit['stages'] if r['original_reference']==original and
                 r['event_prefix_length']==stage['isolation']['event_range'][1]]
        require(len(matches)==1 and matches[0]['immutable_reference']['sha256']==original['sha256'],
                'sampler original stage hash was not reproduced')
        result.update(pool_execution=checked(matches[0]['immutable_reference']),
                      original_pool_execution=original,reconciliation=ref(repair))
    else:
        result['pool_execution']=checked(original)
    return result


def template_render_observations(root, request_path, *, errors=None):
    """Join independently rendered exact exports, never a cross-model reuse claim."""
    root=Path(root).resolve(); request_path=Path(request_path)
    request=read(request_path); request_ref=ref(request_path)
    require(request.get('schema')=='isg-training-template-render-request-v1'
            and request.get('campaign_id')=='campaign_20261110'
            and request.get('production_released') is False
            and request.get('benchmark_credit')==0, 'invalid template render scope')
    results={}
    for record in request['records']:
        try:
            key=(record['candidate_id'],int(record['start']))
            require(key not in results,'duplicate exact template render request')
            directory=Path(record['results']).resolve().parent
            require(directory.is_relative_to(request_path.parent.resolve()),'render output escapes request')
            for name in ('configuration','canary_review','source_environment','native_runtime_identity','manifest','demos'):
                source=record[name]; path=Path(source['path']).resolve()
                require(path.is_relative_to(root) and file_digest(path)==source['sha256'],
                        'template render input changed: '+name)
            row={'request':request_ref,'candidate_id':key[0],'start':key[1],
                 'configuration_sha256':record['configuration_sha256'],
                 'review':record['canary_review'],'native_runtime':record['native_runtime'],
                 'instance_id':record['instance_id'],
                 'status':'prepared','scheduler_state':'unavailable','benchmark_credit':0,
                 'scope':'Independent rendering of this exact sampler export; no cross-model oracle-world reuse assertion.'}
            launch=directory/'launch.json'
            require(launch.is_file(),'template render launch identity missing')
            plan=read(launch)
            require(plan['instances']==[record['instance_id']] and plan['request']==request_ref
                    and plan['configuration']==record['configuration']
                    and plan['native_runtime_identity']==record['native_runtime_identity'],
                    'template render launch differs from request')
            row['launch']=ref(launch)
            submission=directory/'submission.json'
            if submission.exists():
                submitted=read(submission)
                require(submitted.get('request')==request_ref
                        and submitted.get('candidate_id')==key[0]
                        and int(submitted.get('start',-1))==key[1]
                        and submitted.get('native_runtime')==record['native_runtime']
                        and submitted.get('launch')==ref(launch)
                        and submitted.get('source_environment')==record['source_environment'],
                        'template render submission identity differs')
                row.update(status='submitted',submission=ref(submission),
                           job_id=str(submitted['job_id']))
            path=directory/'result.json'
            if path.exists():
                value=read(path)
                require(value['schema']=='isg-render-result-v1'
                        and value['launch_sha256']==file_digest(launch), 'render result launch identity differs')
                iid=record['instance_id'];episodes=value.get('episodes',[])
                passed=(value.get('success') is True and value.get('returncode')==0
                        and value.get('expected_instances')==[iid]
                        and value.get('rendered_instances')==[iid]
                        and value.get('kept_instances')==[iid] and len(episodes)==1
                        and Path(episodes[0].get('demo','')).name==iid
                        and episodes[0].get('kept') is True
                        and episodes[0].get('episode_index') is not None
                        and (episodes[0].get('campaign_restoration')or{}).get('scale_verified') is True)
                row.update(status='verified' if passed else 'failed_validation',
                           result=ref(path),returncode=value.get('returncode'),
                           verification_observed_at=timestamp(path))
                if passed:
                    dataset=Path(episodes[0]['dataset']).resolve()
                    require(dataset.is_relative_to(Path(record['output']).resolve()),'rendered dataset escapes exact output')
                    files=sorted(p for p in dataset.rglob('*')if p.is_file())
                    require(any(p.suffix=='.parquet' for p in files)
                            and any(p.suffix=='.mp4' for p in files), 'render result lacks saved observation data/video')
                    row['artifacts']=[ref(p) for p in files]
            results[key]=row
        except Exception as error:
            if errors is None:
                raise
            errors.append({'path':str(request_path), 'candidate_id':record.get('candidate_id'),
                           'start':record.get('start'), 'error':f'{type(error).__name__}: {error}'})
    return results


_POLICY_SNAPSHOT_CACHE = {}
_POLICY_OBSERVATION_FIELDS = frozenset({'observed_at', 'current_evaluation', 'evaluation_policy_history'})


def policy_snapshot(root, expected):
    """Find hash-exact policy bytes; never reconstruct a historical policy."""
    root=Path(root).resolve()
    require(isinstance(expected,str) and re.fullmatch('[0-9a-f]{64}',expected),
            'review lacks exact gate policy identity')
    key=(str(root),expected)
    cached=_POLICY_SNAPSHOT_CACHE.get(key)
    if cached and _stamp(cached['path'])==cached['stamp']:
        value=read(cached['path'])
        require(_stamp(cached['path'])==cached['stamp'],'cached policy changed during identity check')
        return cached['reference'],value
    paths=[root/'controllers/training_production_gate.json']
    paths.extend(sorted((root/'infra/training_production_gate/current_evaluations_r1').glob('*_previous_policy.json')))
    for path in paths:
        if not path.is_file():continue
        before=_stamp(path)
        if file_digest(path)!=expected:continue
        value=read(path)
        require(_stamp(path)==before,'policy changed during historical identity check')
        reference={'path':str(path),'sha256':expected}
        _POLICY_SNAPSHOT_CACHE[key]={'path':str(path),'stamp':before,'reference':reference}
        return reference,value
    raise ValueError('exact historical gate policy bytes unavailable: '+expected)


def render_review_association(root, row, rendering):
    """Reconcile only observation metadata revisions of identical canary proofs.

    This is reporting evidence, not authorization. Both immutable reviews and
    both exact policy snapshots remain linked. Any authorization/release field,
    source artifact, configuration or runtime difference is rejected.
    """
    root=Path(root).resolve()
    current=(row.get('validated_evidence')or{}).get('review')
    require(rendering['configuration_sha256']==row['configuration_sha256']
            and rendering['native_runtime']==row['native_runtime'] and current,
            'rendering does not belong to this exact validated canary')
    original=rendering['review']
    proofs=[]
    for reference in (original,current):
        path=Path(reference['path']).resolve()
        require(path.is_relative_to(root/'validation/training_canaries/reviews')
                and file_digest(path)==reference['sha256'],'render canary review bytes changed')
        proof=read(path)
        require(proof.get('schema')=='isg-training-canary-review-v1'
                and proof.get('campaign_id')=='campaign_20261110'
                and proof.get('validation_passed') is True and proof.get('accepted') is False
                and proof.get('eligible_for_benchmark') is False and proof.get('quota_consumed')==0,
                'render review is not exact zero-credit evidence')
        require(proof.get('instance_id')==rendering.get('instance_id')
                and proof.get('source_proof',{}).get('config',{}).get('sha256')==row['configuration_sha256'],
                'render review exact instance or configuration differs')
        proofs.append(proof)
    if original==current:return {'status':'exact_review','review':current}
    payloads=[{k:v for k,v in proof.items()if k!='gate_policy_sha256'}for proof in proofs]
    require(payloads[0]==payloads[1],'render review source or validation evidence differs')
    policies=[policy_snapshot(root,proof['gate_policy_sha256'])for proof in proofs]
    semantics=[]
    for reference,policy in policies:
        require(policy.get('schema')=='isg-training-production-gate-v1'
                and policy.get('campaign_id')=='campaign_20261110','render gate policy identity differs')
        if 'evaluation_policy_history'in policy:
            require(policy['evaluation_policy_history'].get('authorization_change') is False,
                    'policy evaluation declared an authorization change')
        if 'current_evaluation'in policy:
            evaluation=policy['current_evaluation'];path=Path(evaluation['path']).resolve()
            require(path.is_relative_to(root/'infra/training_production_gate/current_evaluations_r1')
                    and file_digest(path)==evaluation['sha256'],'policy observation evidence changed')
        semantics.append({k:v for k,v in policy.items()if k not in _POLICY_OBSERVATION_FIELDS})
    require(semantics[0]==semantics[1],'render gate authorization or release semantics differ')
    return {'status':'identical_evidence_policy_observation_revision','render_review':original,
            'current_review':current,'render_policy':policies[0][0],'current_policy':policies[1][0],
            'excluded_policy_metadata':sorted(_POLICY_OBSERVATION_FIELDS),
            'immutable_review_payload_sha256':digest(payloads[0]),
            'policy_semantics_sha256':digest(semantics[0]),'scope':'read-only evidence association; no release authority'}


_BATCH_OBSERVATION_CACHE = {}
_HISTORICAL_RENDER_CACHE = {}


def _historical_render_stamps(root, reference, policy_reference):
    """Watch referenced evidence files, without scanning unrelated code/history."""
    root=Path(root).resolve();watched={};seen=set()
    def visit(value):
        if isinstance(value,dict):
            for child in value.values():visit(child)
        elif isinstance(value,list):
            for child in value:visit(child)
        elif isinstance(value,str) and value.startswith(str(root)+'/'):
            path=Path(value)
            if path in seen:return
            seen.add(path)
            if path.is_dir():return
            watched[str(path)]=_stamp(path)
            if path.is_file() and path.suffix=='.json':visit(read(path))
    visit(reference)
    # The policy file is immutable evidence. Other canaries registered in it
    # are unrelated to this review and must not trigger global history scans.
    watched[policy_reference['path']]=_stamp(policy_reference['path'])
    return watched


def historical_render_association(root, row, rendering, *, auditor=None):
    """Validate a finished render against its original policy, not current permission."""
    root=Path(root).resolve();original=rendering['review']
    exact_row={**row,'validated_evidence':{'review':original}}
    render_review_association(root,exact_row,rendering)
    proof=read(original['path']);policy_reference,_=policy_snapshot(root,proof['gate_policy_sha256'])
    key=(str(root),original['path'],original['sha256'],policy_reference['sha256'])
    cached=_HISTORICAL_RENDER_CACHE.get(key)
    if cached and all(_stamp(path)==stamp for path,stamp in cached['stamps'].items()):
        verification={**cached['verification'],'verification_cached':True}
    else:
        if auditor is None:
            from scripts.campaign_20261110.training_canaries import audit_historical_canary
            auditor=audit_historical_canary
        before=_historical_render_stamps(root,original,policy_reference)
        actual=auditor(proof['source_job'],proof['source_proof']['source_gate']['path'],
                       root=root,started_at=1794321600,policy_reference=policy_reference)
        require(digest(actual)==digest(proof),'historical canary physical or authorization evidence changed')
        require(all(_stamp(path)==stamp for path,stamp in before.items()),
                'historical canary evidence changed during audit')
        verification={'verification_observed_at':datetime.now(timezone.utc).isoformat(),
                      'verification_cached':False,'full_review_payload_sha256':digest(actual)}
        _HISTORICAL_RENDER_CACHE[key]={'verification':verification,'stamps':before}
    current=(row.get('validated_evidence')or{}).get('review')
    result={'status':'historical_review_reaudited','render_review':original,
            'historical_policy':policy_reference,'current_review':current,
            'authorization_equivalence_asserted':False,'current_permission_from_this_evidence':False,
            'scope':'Completed render evidence under its exact original policy; current authorization is separate.',
            **verification}
    if current:
        path=Path(current['path']).resolve()
        require(path.is_relative_to(root/'validation/training_canaries/reviews')
                and file_digest(path)==current['sha256'],'current canary review bytes changed')
        current_proof=read(path)
        require({k:v for k,v in proof.items()if k!='gate_policy_sha256'}==
                {k:v for k,v in current_proof.items()if k!='gate_policy_sha256'},
                'current canary source or validation evidence differs from completed render')
        result['current_review_has_identical_immutable_evidence']=True
        result['current_policy_sha256']=current_proof['gate_policy_sha256']
    return result


def _evidence_stamps(root, probe):
    """Track transitive proof files and frozen source trees without rehashing videos.

    This is a read-only reporting cache. Release verifiers continue to hash and
    audit the original artifacts directly; they never consume this cache.
    """
    root=Path(root).resolve(); watched=set(); visited=set(); seen=set()
    def visit(value):
        if isinstance(value,dict):
            for item in value.values(): visit(item)
        elif isinstance(value,list):
            for item in value: visit(item)
        elif isinstance(value,str) and value.startswith(str(root)+'/'):
            path=Path(value)
            if path in seen: return
            seen.add(path)
            if not path.exists():
                watched.add(path)
                return
            if path.is_dir():
                if path.parent==root/'code':
                    watched.add(path/'.snapshot_commit')
                    for folder in ('ispatialgym','scripts'):
                        watched.add(path/folder)
                        watched.update(p for p in (path/folder).rglob('*') if p.is_dir() or p.suffix=='.py')
                return
            watched.add(path)
            if path.suffix=='.json' and path not in visited:
                visited.add(path); visit(read(path))
    visit(str(probe))
    # The compatibility proof names its base by commit instead of by path.
    for path in list(visited):
        value=read(path)
        if isinstance(value,dict) and value.get('schema')=='isg-observational-runtime-compatibility-v1':
            for key in ('base_commit','observational_commit'):
                visit(str(root/'code'/('trunk_'+value[key][:12])))
    return {str(path):_stamp(path) for path in watched}


def _stamp(path):
    try:
        s=Path(path).stat()
        return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode)
    except FileNotFoundError:
        return None


def observed_batch_verification(root, reference, kind):
    """Reuse a dated audit only while every observed input identity is unchanged."""
    key=(str(Path(root).resolve()),reference['path'],reference['sha256'],kind)
    previous=_BATCH_OBSERVATION_CACHE.get(key)
    if previous and all(_stamp(path)==stamp for path,stamp in previous['stamps'].items()):
        return {**previous['result'],'verification_cached':True}
    from scripts.campaign_20261110.training_restoration import verify_batch
    before=_evidence_stamps(root,reference['path'])
    result={'verification_observed_at':datetime.now(timezone.utc).isoformat(),
            'verification_cached':False,'verification_scope':'reporting only; release rechecks original evidence'}
    try:
        verify_batch(root,reference,1794321600,kind)
        result['status']='verified'
    except Exception as error:
        result.update(status='failed_validation',error=f'{type(error).__name__}: {error}')
    after={path:_stamp(path) for path in before}
    require(after==before,'batch evidence changed during observation')
    result['verification_finished_at']=datetime.now(timezone.utc).isoformat()
    _BATCH_OBSERVATION_CACHE[key]={'stamps':after,'result':result}
    return result


class BatchObservationWorker:
    """Bounded read-only audit worker; slow proofs never block instance polling."""
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='batch-proof-report')
        self.work = {}

    def observe(self, root, reference, kind):
        key=(str(Path(root).resolve()),reference['path'],reference['sha256'],kind)
        entry=self.work.setdefault(key,{})
        future=entry.get('future')
        if future is not None and future.done():
            try: entry['result']=future.result()
            except Exception as error:
                entry['result']={'status':'verification_unavailable','error':f'{type(error).__name__}: {error}'}
            entry.update(future=None,finished=time.monotonic())
        if entry.get('future') is None and time.monotonic()-entry.get('finished',-float('inf'))>=30:
            entry['future']=self.executor.submit(observed_batch_verification,root,reference,kind)
        return {**entry.get('result',{'status':'verification_pending'}),
                'verification_refresh_pending':entry.get('future') is not None,
                'verification_scope':'read-only reporting; release rechecks original evidence'}


def observe(root, *, now=None, publish=None, previous_batches=(), batch_verifier=observed_batch_verification):
    root = Path(root).resolve(); base = root/'training_pipeline_canaries'
    result = {'schema': 'isg-training-pipeline-status-v1', 'campaign_id': 'campaign_20261110',
        'observed_at': now or datetime.now(timezone.utc).isoformat(), 'records': [], 'batch_stages': [],
        'sources': [], 'errors': [], 'diagnostic_controls': [], 'benchmark_credit': 0, 'quota_consumed': 0,
        'production_released': False, 'status': 'observed'}
    result['cycle_started_at'] = result['observed_at']
    result['cycle_finished_at'] = None
    def checkpoint(phase):
        if publish is None: return
        current = {b['request']['path'] for b in result['batch_stages']}
        carried = [{**b, 'observation_current_cycle': False} for b in previous_batches
                   if b.get('request', {}).get('path') not in current]
        snapshot = {**result, 'batch_stages': result['batch_stages'] + carried,
                    'cycle_phase': phase, 'status': 'refreshing'}
        summarize(snapshot)
        publish(snapshot)
    reviews = {}
    renders = {}
    for path in sorted(base.glob('*/template_rendering*/request.json')):
        try:
            observed=template_render_observations(root,path,errors=result['errors'])
            scoped={(str(path.parent.parent.resolve()), *key):row for key,row in observed.items()}
            require(not(set(scoped)&set(renders)), 'duplicate rendering attempt for exact canary')
            renders.update(scoped)
            result['sources'].append({**ref(path),'kind':'exact_template_rendering'})
        except Exception as error:
            result['errors'].append({'path':str(path),'error':f'{type(error).__name__}: {error}'})
    for path in sorted(base.glob('*/existing_native_reviews.json')):
        try:
            value = read(path)
            require(value['campaign_id'] == result['campaign_id'], 'foreign review observation')
            result['sources'].append({**ref(path), 'observed_at': value.get('observed_at'), 'kind': 'existing_native_reviews'})
            for candidate, row in value['records'].items():
                allocation = row.get('allocation') or {}
                start = allocation.get('start')
                if start is not None:
                    reviews[(str(path.parent.resolve()), candidate, int(start))] = (row, value.get('observed_at'), ref(path))
        except Exception as error:
            result['errors'].append({'path': str(path), 'error': f'{type(error).__name__}: {error}'})
    for path in sorted(base.glob('*/requests.json')):
        try:
            value = read(path)
            require(value['campaign_id'] == result['campaign_id'], 'foreign canary request')
            original_request = ref(path)
            current_request = original_request
            pointer_path = path.parent/'current_request.json'
            if pointer_path.exists():
                pointer = read(pointer_path)
                require(pointer['previous'] == original_request, 'request revision original differs')
                current_path = Path(pointer['current']['path']).resolve()
                require(current_path.parent == path.parent.resolve() and
                        file_digest(current_path) == pointer['current']['sha256'], 'request revision changed')
                revised = read(current_path)
                mutable = {'authorization_path', 'authorization_sha256', 'expires_at', 'status'}
                require(revised['campaign_id'] == value['campaign_id'] and
                        [{k:v for k,v in r.items() if k not in mutable} for r in revised['records']] ==
                        [{k:v for k,v in r.items() if k not in mutable} for r in value['records']],
                        'authorization refresh changed native attempt identity')
                value, current_request = revised, pointer['current']
            for request in value['records']:
                row = None
                try:
                    key = (request['candidate_id'], int(request['start']))
                    row = {'candidate_id': key[0], 'start': key[1], 'count': request['count'],
                        'class_id': request['class_id'], 'configuration_sha256': request['config_sha256'],
                        'native_runtime': request['native_runtime'], 'batch': path.parent.name,
                        'request': current_request, 'original_request': original_request,
                        'status': 'native_not_observed', 'native': None,
                        'review': None, 'benchmark_credit': 0, 'quota_consumed': 0, 'dependencies': []}
                    config_path = request.get('config_path')
                    if config_path:
                        config_path = Path(config_path).resolve()
                        require(config_path.is_relative_to(root) and file_digest(config_path)==request['config_sha256'],
                                'native configuration evidence changed')
                        row['task'] = read(config_path)['template']['task']
                        row['configuration'] = ref(config_path)
                    for name,sha_name in [('config_path','config_sha256'),('input_manifest','input_manifest_sha256'),('authorization_path','authorization_sha256')]:
                        if not request.get(name):continue
                        artifact=Path(request[name]).resolve()
                        require(artifact.is_relative_to(root) and file_digest(artifact)==request[sha_name],
                                'canary prerequisite identity changed: '+name)
                        row['dependencies'].append({'id':name+':'+request[sha_name],
                            'required_for':['native_generation','collection','physics_replay'],
                            'state':'verified','scope':'immutable input identity; physical runtime checks remain mandatory',
                            'artifact':ref(artifact),'source_ref':current_request})
                    output = Path(request['output']).resolve()
                    require(output.is_relative_to(base), 'native output outside pipeline scope')
                    submitted_path = path.parent/'submission.json'
                    submitted_pointer = path.parent/'submission_current.json'
                    if submitted_pointer.exists():
                        pointer = read(submitted_pointer)
                        selected_ref = pointer.get('current', pointer)
                        selected = Path(selected_ref['path']).resolve()
                        require(selected.parent == path.parent.resolve() and
                                file_digest(selected) == selected_ref['sha256'], 'native submission revision changed')
                        row['native_submission_pointer'] = ref(submitted_pointer)
                        if submitted_path.exists(): row['original_native_submission'] = ref(submitted_path)
                        submitted_path = selected
                    if submitted_path.exists():
                        submitted = read(submitted_path)
                        if 'jobs' in submitted:
                            require(submitted['request'] in (current_request,original_request),
                                    'native submission request differs')
                            matches=[job for job in submitted['jobs'] if job.get('candidate_id')==key[0]]
                            require(len(matches)<=1,'duplicate native member submission')
                            submitted=matches[0] if matches else {}
                            if submitted:
                                require(submitted.get('config_sha256')==request['config_sha256'] and
                                        submitted.get('count')==request['count'],
                                        'native member configuration/count differs')
                        if all(k in submitted for k in ('job_id','native_runtime','request')) and ('start' in submitted or 'members' in submitted):
                            require(submitted['request'] in (current_request,original_request)
                                    and submitted['native_runtime']==request['native_runtime']
                                    and (int(submitted['start'])==key[1] if 'start' in submitted else
                                         sum(m.get('candidate_id')==key[0] and int(m.get('start',-1))==key[1]
                                             and m.get('config_sha256')==request['config_sha256']
                                             and m.get('count')==request['count'] for m in submitted['members'])==1)
                                    and submitted.get('returncode')==0,
                                    'native submission does not bind this exact request/runtime/index')
                            row['native_submission'] = {'evidence':ref(submitted_path),
                                'job_id':str(submitted['job_id']),'observed_at':submitted.get('observed_at'),
                                'status':'submitted','scheduler_state':'unavailable'}
                            row['status'] = 'native_submitted'
                    gate_path = output/'sampling_gate.json'
                    if gate_path.exists():
                        gate = read(gate_path)
                        require(gate['candidate_id'] == key[0] and gate['config_sha256'] == request['config_sha256'],
                                'native gate configuration differs')
                        row['native'] = {**ref(gate_path), 'observed_at': timestamp(gate_path),
                            'returncode': gate.get('returncode'), 'written': gate.get('tro_count'),
                            'elapsed_seconds': gate.get('elapsed_seconds')}
                        row['status'] = 'native_passed' if gate.get('returncode') == 0 and gate.get('tro_count') == 1 else 'native_failed'
                        if row['status']=='native_passed':
                            from scripts.campaign_20261110.training_stream import native_evidence
                            row['native_proof']=native_evidence(request,root)
                            row['dependencies'].append({'id':'native_gate:'+file_digest(gate_path),
                                'required_for':['collection','physics_replay'],'state':'verified',
                                'artifact':ref(gate_path),'source_ref':current_request,
                                'scope':'this exact index/config/runtime and hash-checked native TRO only'})
                    scoped_key=(str(path.parent.resolve()), *key)
                    if scoped_key in reviews:
                        record, observed_at, source = reviews[scoped_key]
                        row['review'] = {**record, 'observed_at': observed_at, 'observation_source': source}
                        row['status'] = record['status']
                        if record.get('validation_review'):
                            validation = record['validation_review']; review_path = Path(validation['review']).resolve()
                            require(review_path.is_relative_to(root) and file_digest(review_path) == validation['sha256'],
                                    'canary review changed')
                            proof = read(review_path)
                            require(proof['validation_passed'] is True and proof['accepted'] is False and
                                    proof['quota_consumed'] == 0, 'review is not zero-credit canary evidence')
                            row['task'] = proof['export_record']['task']
                            row['validated_evidence'] = {'review': ref(review_path),
                                'validation_key': proof['export_record']['validation_key'],
                                'scope': 'this exact configuration and instance only',
                                'native': True, 'demo': True, 'physics_replay': True, 'hard_qa': True,
                                'export_provenance': True, 'benchmark_acceptance': False,
                                'perturbations': proof['export_record'].get('perturbations', {})}
                    if value.get('sequence') == ['A','B','A']:
                        row['sampler_sequence']={'sequence':['A','B','A'],'request':current_request,
                            'status':'prepared','scope':'Same loaded native SAMPLER world and active-only exports; each exact demo/replay/render remains separate.',
                            'benchmark_credit':0,'release_eligible':False}
                        sequence_path=path.parent/'sequence_result.json'
                        if sequence_path.exists():
                            sequence=read(sequence_path)
                            require(sequence['request']==current_request and sequence['benchmark_credit']==0
                                    and sequence['production_released'] is False,'sampler sequence identity differs')
                            stages=[x for x in sequence['stages'] if x['candidate_id']==key[0] and int(x['start'])==key[1]]
                            require(len(stages)<=1,'duplicate sequence stage')
                            row['sampler_sequence'].update(status=sequence['status'],evidence=ref(sequence_path),
                                observed_at=sequence['observed_at'],stage=stages[0] if stages else None)
                            if stages:
                                row['sampler_sequence']['stage_references']=sequence_stage_references(root,sequence_path,stages[0])
                                row['sampler_sequence']['evidence_identity_verified']=True
                    if scoped_key in renders:
                        rendering=renders[scoped_key]
                        association=historical_render_association(root,row,rendering)
                        row['template_rendering']={**rendering,'review_association':association}
                        if rendering['status']=='verified':
                            target='validated_evidence' if row.get('validated_evidence') else 'historical_validated_evidence'
                            row.setdefault(target,{'review':rendering['review'],'scope':'Historical exact canary; no current permission'})
                            row[target]['rendering']=True
                            row[target]['rendering_result']=rendering['result']
                    row['last_observed_at'] = datetime.now(timezone.utc).isoformat()
                    result['records'].append(row)
                except Exception as error:
                    detail={'path':str(path), 'candidate_id':request.get('candidate_id'),
                            'start':request.get('start'), 'error':f'{type(error).__name__}: {error}'}
                    result['errors'].append(detail)
                    if row is not None:
                        row.update(status='evidence_identity_failed', observation_error=detail,
                                   last_observed_at=datetime.now(timezone.utc).isoformat())
                        result['records'].append(row)
        except Exception as error:
            result['errors'].append({'path': str(path), 'error': f'{type(error).__name__}: {error}'})
    controls=set(base.glob('*/restoration_control*/request.json'))
    controls.update(base.glob('*/batch_restoration*/control_aa/request.json'))
    for request in sorted(controls):
        try:
            control=read(request)
            require(control.get('diagnostic_control')=='A_A' and control.get('diagnostic_only') is True
                    and control.get('production_released') is False,'invalid reset control request')
            row={'kind':'repeated_A_physics_control','request':ref(request),'status':'prepared',
                 'observed_at':timestamp(request),'benchmark_credit':0,'release_eligible':False,
                 'scope':'Diagnostic reset-variability baseline only; cannot satisfy an A/B/A release gate.'}
            submission=request.parent/'submission.json'
            if submission.exists():
                submitted=read(submission);require(submitted['request']==ref(request),'control submission identity differs')
                row.update(status='submitted',submission=ref(submission),allocations=submitted['jobs'],scheduler_state='unavailable')
            probe=request.parent/'physics/probe.json'
            if probe.exists():
                observed=read(probe)
                require(observed['schema']=='isg-training-restoration-control-v1' and observed['request']==ref(request),
                        'control probe identity differs')
                entries=observed['entries'];require(len(entries)<=2,'control episode bound exceeded')
                row.update(status='episodes_recorded',probe=ref(probe),episodes_observed=len(entries),
                           observed_at=timestamp(probe),successful_episodes=sum(e.get('returncode')==0 and
                           e.get('result',{}).get('success') is True for e in entries))
            result['diagnostic_controls'].append(row)
        except Exception as error:
            result['errors'].append({'path':str(request),'error':f'{type(error).__name__}: {error}'})
    # Publish freshly joined instance outcomes before any potentially slow full
    # batch proof. Prior batch observations keep their own dated evidence.
    checkpoint('batch_verification')
    for request in sorted(base.glob('*/batch_restoration*/request.json')):
        batch = {'request': ref(request), 'batch': str(request.parent.relative_to(base)),
                 'observed_at': timestamp(request), 'status': 'prepared_unvalidated', 'stages': [],
                 'benchmark_credit': 0, 'quota_consumed': 0}
        batch['tasks'] = []
        for source in read(request).get('canaries', []):
            try:
                source_path = Path(source['path']).resolve()
                require(source_path.is_relative_to(root) and file_digest(source_path)==source['sha256'],
                        'batch canary review changed')
                task = read(source_path)['export_record']['task']
                if task not in batch['tasks']: batch['tasks'].append(task)
            except Exception as error:
                batch.setdefault('source_errors', []).append(str(error))
        failure = request.parent/'preflight_failure.json'
        if failure.exists(): batch.update(status='preflight_failed', failure={**ref(failure), **read(failure)})
        elif (request.parent/'preflight.json').exists(): batch['status'] = 'ready_for_bounded_probe'
        submission_path = request.parent/'submission.json'
        submissions = {}
        submission_pointer = request.parent/'submission_current.json'
        if submission_pointer.exists():
            try:
                pointer = read(submission_pointer)
                selected = Path(pointer['current']['path']).resolve()
                require(selected.parent == request.parent.resolve() and
                        file_digest(selected) == pointer['current']['sha256'], 'submission revision changed')
                submission_path = selected
                batch['prior_attempts'] = pointer['prior_attempts']
                batch['submission_pointer'] = ref(submission_pointer)
            except Exception as error:
                batch.update(status='submission_observation_failed', error=f'{type(error).__name__}: {error}')
        if submission_path.exists():
            try:
                submitted = read(submission_path)
                require(submitted['request'] == ref(request), 'batch submission request differs')
                submissions = {job['kind']: job for job in submitted['jobs']}
                batch['submission'] = ref(submission_path)
            except Exception as error:
                batch.update(status='submission_observation_failed', error=f'{type(error).__name__}: {error}')
        scheduler_path=request.parent/'execution_observation.json'
        scheduler={}
        if scheduler_path.exists():
            try:
                observation=read(scheduler_path)
                require(observation['schema']=='isg-batch-execution-observation-v1' and
                        observation['request']==ref(request),'batch execution observation identity differs')
                scheduler={str(job['job_id']):job for job in observation['jobs']}
                batch['execution_observation']=ref(scheduler_path)
            except Exception as error:
                batch.setdefault('observation_errors',[]).append(str(error))
        for name, kind in [('physics', 'physics_action_open_loop'), ('render', 'kinematic_observation_rendering')]:
            probe = request.parent/name/'probe.json'
            stage = {'kind': kind, 'status': 'not_observed'}
            if kind in submissions:
                stage.update(status='submitted', allocation=submissions[kind], scheduler_state='unavailable')
            native_failure = request.parent/name/'failure.json'
            if native_failure.exists():
                stage.update(status='failed_execution', failure=ref(native_failure),
                             error=read(native_failure).get('error'), observed_at=timestamp(native_failure))
            if probe.exists():
                stage.update(evidence=ref(probe), observed_at=timestamp(probe), status='incomplete')
                try:
                    native = read(probe); stage['episodes_observed'] = len(native.get('entries', []))
                    if any(entry.get('returncode', 0)!=0 or entry.get('result', {}).get('success') is False
                           for entry in native.get('entries', [])):
                        stage['status'] = 'failed_execution'
                    if stage['episodes_observed'] == 3:
                        stage.update(batch_verifier(root,ref(probe),kind))
                        if stage['status'] in ('verification_pending','verification_unavailable'):
                            prior=next((s for b in previous_batches if b.get('request')==batch['request']
                                        for s in b.get('stages',[]) if s.get('kind')==kind
                                        and s.get('evidence')==stage['evidence']),None)
                            if prior:
                                history=prior.get('historical_verification')
                                if prior.get('status') in ('verified','failed_validation'):
                                    history={k:prior.get(k) for k in ('status','error','evidence',
                                             'verification_observed_at','verification_finished_at')}
                                if history: stage['historical_verification']=history
                except Exception as error:
                    stage.update(status='failed_validation', error=f'{type(error).__name__}: {error}')
            job_id=str(submissions.get(kind,{}).get('job_id',''))
            if job_id in scheduler:
                execution=scheduler[job_id]
                stage.update(scheduler_state=execution['state'],scheduler_observed_at=observation['observed_at'],
                             execution_observation=ref(scheduler_path),exit_code=execution.get('exit_code'))
                if execution['state'] in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'):
                    stage['evidence_verified']=stage['status']=='verified'
                    stage['status']='failed_execution'
                elif execution['state']=='COMPLETED' and stage['status'] in ('submitted','incomplete'):
                    stage['status']='completed_without_complete_probe'
            batch['stages'].append(stage)
        if all(s['status'] == 'verified' for s in batch['stages']): batch['status'] = 'batch_restoration_verified'
        elif any(s['status'] in ('failed_execution','failed_validation')for s in batch['stages']):batch['status']='failed_validation'
        elif any(s['status']=='completed_without_complete_probe'for s in batch['stages']):batch['status']='incomplete_terminal_evidence'
        elif any(s['status'] in ('verification_pending','verification_unavailable')for s in batch['stages']):batch['status']='verification_pending'
        elif submissions:batch['status']='submitted'
        batch['observation_current_cycle'] = True
        batch['last_observed_at'] = datetime.now(timezone.utc).isoformat()
        result['batch_stages'].append(batch)
        checkpoint('batch_verification')
    result['cycle_finished_at'] = datetime.now(timezone.utc).isoformat()
    result['cycle_phase'] = 'complete'
    summarize(result)
    return result


def summarize(result):
    keys = [(r['original_request']['path'], r['candidate_id'], r['start']) for r in result['records']]
    require(len(keys) == len(set(keys)), 'duplicate exact canary attempt in projection')
    result['counts'] = dict(Counter(r['status'] for r in result['records']))
    result['attempts_observed'] = len(result['records'])
    result['family_paths'] = family_paths(result['records'], result['batch_stages'])
    if result['errors']: result['status'] = 'partial_observation'



def family_paths(records, batches):
    """All23 paths remain explicit; one exact canary never validates a family."""
    result=[]
    for quota in task_quotas():
        selected=[r for r in records if r.get('task')==quota.task]
        passed=[r for r in selected if r.get('validated_evidence')]
        evidence=[r['validated_evidence']['review'] for r in passed]
        sequences=[r['sampler_sequence'] for r in selected if r.get('sampler_sequence',{}).get('evidence_identity_verified')
                   and r['sampler_sequence']['status']=='sampler_sequence_passed'
                   and r['sampler_sequence']['stage'].get('native_passed') is True]
        paths={}
        for kind in ('physics_action_open_loop','kinematic_observation_rendering'):
            stages=[(batch,stage) for batch in batches if quota.task in batch.get('tasks',[])
                    for stage in batch['stages'] if stage['kind']==kind]
            verified=[stage['evidence'] for _,stage in stages if stage['status']=='verified']
            paths[kind]={'status':'partial' if verified else
                'failed' if any(stage['status'] in ('failed_execution','failed_validation') for _,stage in stages) else
                'pending' if stages else 'untested',
                'verified_exact_probes':verified,'attempts':[{'request':b['request'],**stage}for b,stage in stages],
                'owner':'/root/dataset_contract','scope':'only the referenced configurations; no whole-family inference'}
        result.append({'task':quota.task,'standard_classes':quota.standard_classes,
            'native_attempts_observed':len(selected),'native_written':sum((r.get('native')or{}).get('written')or 0 for r in selected),
            'exact_configurations_with_demo_physics_export':len({r['configuration_sha256']for r in passed}),
            'exact_validation':{'status':'partial' if passed else 'untested','evidence':evidence},
            'exact_template_rendering':{'status':'partial' if any(r.get('template_rendering',{}).get('status')=='verified'for r in selected) else 'pending' if any(r.get('template_rendering')for r in selected) else 'untested',
                'attempts':[r['template_rendering']for r in selected if r.get('template_rendering')],
                'scope':'Only independently rendered exact templates; no global or cross-model oracle-world reuse conclusion.'},
            'aba_restoration':paths,
            'model_pool_switch_active_only_export':{'status':'partial' if sequences else 'untested','owner':'/root/dataset_contract',
                'exact_native_stages':sequences,
                'scope':'Only referenced sampler-world native switching/export evidence; independent exact template demo/replay/render remain separate.',
                'next_action':'Complete each exact template demo/replay/render and selected configuration coverage; no whole-family inference.'},
            'required_configuration_denominator':None,'denominator_status':'owned_full_selected_inventory_in_progress',
            'dependency_owner':'/root/templates_assets','runtime_owner':'/root',
            'next_action':'Advance each dependency-ready exact configuration independently; validate this family path with bounded zero-credit canaries.',
            'production_released':False})
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--interval', type=float, default=0)
    args = parser.parse_args(); output = args.root/'training_pipeline_canaries/pipeline_status.json'
    worker=BatchObservationWorker() if args.interval>0 else None
    while True:
        previous = read(output).get('batch_stages', []) if output.exists() else []
        value = observe(args.root, publish=lambda value: write_json_atomic(output, value),
                        previous_batches=previous,
                        batch_verifier=worker.observe if worker else observed_batch_verification)
        write_json_atomic(output, value)
        print(json.dumps({'observed_at': value['observed_at'], 'attempts': value['attempts_observed'],
            'errors': len(value['errors']), 'benchmark_credit': 0}), flush=True)
        if args.interval <= 0: break
        time.sleep(args.interval)


if __name__ == '__main__': main()
