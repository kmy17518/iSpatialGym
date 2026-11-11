"""Read-only projection of bounded pipeline attempts; never author or submit work."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from scripts.campaign_20261110.contracts import file_digest, require, write_json_atomic, task_quotas


def read(path): return json.loads(Path(path).read_text())
def ref(path): return {'path': str(path), 'sha256': file_digest(path)}
def timestamp(path): return datetime.fromtimestamp(Path(path).stat().st_mtime, timezone.utc).isoformat()


_BATCH_OBSERVATION_CACHE = {}


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
    _BATCH_OBSERVATION_CACHE[key]={'stamps':after,'result':result}
    return result


def observe(root, *, now=None, publish=None, previous_batches=()):
    root = Path(root).resolve(); base = root/'training_pipeline_canaries'
    result = {'schema': 'isg-training-pipeline-status-v1', 'campaign_id': 'campaign_20261110',
        'observed_at': now or datetime.now(timezone.utc).isoformat(), 'records': [], 'batch_stages': [],
        'sources': [], 'errors': [], 'benchmark_credit': 0, 'quota_consumed': 0,
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
    for path in sorted(base.glob('*/existing_native_reviews.json')):
        try:
            value = read(path)
            require(value['campaign_id'] == result['campaign_id'], 'foreign review observation')
            result['sources'].append({**ref(path), 'observed_at': value.get('observed_at'), 'kind': 'existing_native_reviews'})
            for candidate, row in value['records'].items():
                allocation = row.get('allocation') or {}
                start = allocation.get('start')
                if start is not None:
                    reviews[(candidate, int(start))] = (row, value.get('observed_at'), ref(path))
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
                if submitted_path.exists():
                    submitted = read(submitted_path)
                    if all(k in submitted for k in ('job_id','start','native_runtime','request')):
                        require(submitted['request'] in (current_request,original_request)
                                and submitted['native_runtime']==request['native_runtime']
                                and int(submitted['start'])==key[1]
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
                if key in reviews:
                    record, observed_at, source = reviews[key]
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
                row['last_observed_at'] = datetime.now(timezone.utc).isoformat()
                result['records'].append(row)
        except Exception as error:
            result['errors'].append({'path': str(path), 'error': f'{type(error).__name__}: {error}'})
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
                        stage.update(observed_batch_verification(root,ref(probe),kind))
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
    keys = [(r['candidate_id'], r['start']) for r in result['records']]
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
            'aba_restoration':paths,
            'model_pool_switch_active_only_export':{'status':'untested','owner':'/root',
                'next_action':'Execute different active model selections in a reused preloaded world and audit exports; A/B/A is separate.'},
            'required_configuration_denominator':None,'denominator_status':'owned_full_selected_inventory_in_progress',
            'dependency_owner':'/root/templates_assets','runtime_owner':'/root',
            'next_action':'Advance each dependency-ready exact configuration independently; validate this family path with bounded zero-credit canaries.',
            'production_released':False})
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--interval', type=float, default=0)
    args = parser.parse_args(); output = args.root/'training_pipeline_canaries/pipeline_status.json'
    while True:
        previous = read(output).get('batch_stages', []) if output.exists() else []
        value = observe(args.root, publish=lambda value: write_json_atomic(output, value),
                        previous_batches=previous)
        write_json_atomic(output, value)
        print(json.dumps({'observed_at': value['observed_at'], 'attempts': value['attempts_observed'],
            'errors': len(value['errors']), 'benchmark_credit': 0}), flush=True)
        if args.interval <= 0: break
        time.sleep(args.interval)


if __name__ == '__main__': main()
