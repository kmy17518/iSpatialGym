"""Route exact, unclaimed floating feasibility queries through existing Vast transport.

Cluster and Vast share physical claims. This adapter only changes transport,
output paths and reserved indices; task inputs and bounded sampling stay pinned.
"""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import sys
import time

from scripts.campaign_20261110 import vast_native_dependency as native
from scripts.campaign_20261110.planning_indices import prepare as reserve
from scripts.campaign_20261110.production_gate import check_job


def normalized_entry(query):
    native.require(query['production_authorized'] is False and query['benchmark_credit'] == 0,
                   'Only zero-credit physical preparation is authorized')
    native.require(query['native_time_seconds'] == 1800 and query['hard_time_seconds'] == 2400,
                   'Unexpected query bounds')
    argv = list(query['sample_argv'])
    native.require(argv[:4] == ['python', '-u', '-m', 'ispatialgym.sampling.sample_isg'],
                   'Unexpected native entry point')
    argv.pop(1)  # Buffering only; the helper runs the same Python module.
    native.require('--start-index' not in argv and '--time-budget-s' not in argv,
                   'Query already has a different reservation or budget')
    for flag, expected in [('--max-attempts', '3'), ('--templates', '1'),
                           ('--instances-per-template', '1'), ('--out', '${NATIVE_OUTPUT}'),
                           ('--scene', query['scene']), ('--room', query['room'])]:
        native.require(argv.count(flag) == 1 and argv[argv.index(flag)+1] == expected,
                       'Query contract differs: '+flag)
    native.require('--no-write-back' in argv, 'Query must preserve source inputs')
    camera=query['class_id'].startswith('camera_relocalization-')
    environment=query['environment'] if camera else {}
    if camera:
        native.require(environment=={'ISPATIALGYM_CAMPAIGN_OBSTACLE_GOAL':'1'}
                       and query['class_id'].startswith('camera_relocalization-obstructed-'),
                       'Camera transport requires the exact stationary-obstacle goal')
    family='camera_feasibility' if camera else 'floating_feasibility'
    argv[argv.index('--out')+1] = '${CAMPAIGN_OUT}/'+family+'/'+query['id']
    return {'class_id':query['class_id'], 'scene':query['scene'], 'environment':environment,
            'sample_argv':argv, 'production_released':False,
            'source_query_sha256':query['request']['sha256'],
            'source_query_id':query['id'], 'physical_dedup_key':query['physical_dedup_key']}


def immutable_write(path, value):
    if path.exists():
        native.require(json.loads(path.read_text()) == value, 'Prepared transport changed: '+str(path))
    else:
        native.write(path, value)


def run(root, manifest, expected_sha, helper, *, leave_for_cluster, transport_name='vast_transport', max_new_jobs=None):
    root, manifest, helper = map(lambda p:Path(p).resolve(), (root, manifest, helper))
    native.require(native.sha(manifest) == expected_sha, 'Source manifest changed')
    source = json.loads(manifest.read_text())
    native.require(source['campaign_id'] == root.name and source['benchmark_credit'] == 0
                   and source['production_authorized'] is False, 'Source purpose differs')
    camera=all(q['class_id'].startswith('camera_relocalization-') for q in source['jobs'])
    native.require(transport_name.replace('_','').isalnum(),'Invalid transport directory name')
    family='camera_feasibility' if camera else 'floating_feasibility'
    namespace='camera_query_execution_r1' if camera else 'floating_feasibility_execution_r1'
    directory = root/'training_preparation/model_variant_inventory_r2'/namespace
    transport = directory/transport_name;transport.mkdir(parents=True, exist_ok=True)
    for name in ('claims', 'receipts'):
        (directory/name).mkdir(parents=True, exist_ok=True)
    queries = source['jobs']
    reservation = reserve(root, 'floating_feasibility_vast_r1_'+expected_sha[:16], len(queries))
    digest_cache = {}
    def checked(ref):
        name = ref['path']
        if name not in digest_cache:digest_cache[name] = native.sha(name)
        native.require(digest_cache[name] == ref['sha256'], 'Source input changed: '+name)
    progress_path=transport/'progress.json'
    previous=json.loads(progress_path.read_text()) if progress_path.exists() else {}
    source_ref={'path':str(manifest),'sha256':expected_sha}
    if previous.get('source'):
        native.require(previous['source']==source_ref,'Transport source changed')
    releases = {};submitted = list(previous.get('submitted',[]));skipped = [];errors = []
    added=0
    for index, query in enumerate(queries):
        if max_new_jobs is not None and added>=max_new_jobs:break
        physical = query['physical_dedup_key']
        claim = directory/'claims'/(physical+'.json');receipt = directory/'receipts'/(physical+'.json')
        if index < leave_for_cluster or claim.exists() or receipt.exists():
            skipped.append({'id':query['id'], 'reason':'cluster prefix or existing physical ownership'});continue
        try:
            entry = normalized_entry(query)
            for ref in query['input_refs']+[query['request']]:checked(ref)
            if camera:
                original_release=transport/'camera_no_grasp_dependency'
                for part in ('glib/models','maps'):
                    p=original_release/part;p.mkdir(parents=True,exist_ok=True)
                    (p/'.unused_by_camera').touch(exist_ok=True)
                immutable_write(original_release/'anchors.json',{})
            else:original_release = Path(query['dependency_release'])
            if str(original_release) not in releases:
                release = transport/'releases'/original_release.name
                if not release.exists():shutil.copytree(original_release, release)
                files = {}
                for path in original_release.rglob('*'):
                    if path.is_file():
                        relative = str(path.relative_to(original_release));digest=native.sha(path)
                        native.require(native.sha(release/relative)==digest, 'Library copy differs')
                        files[relative]=digest
                native.require('anchors.json' in files and (camera or any(n.endswith('/library.json') for n in files)),
                               'Incomplete exact library release')
                release_manifest=release/'release_manifest.json'
                if release_manifest.exists():
                    prior=json.loads(release_manifest.read_text())
                    native.require(prior['files']==files,'Existing library release changed')
                    checked(prior['source'])
                else:immutable_write(release_manifest, {'files':files, 'source':query['request']})
                releases[str(original_release)] = release
            release = releases[str(original_release)]
            source_dir = transport/'queries'/query['id'];source_dir.mkdir(parents=True, exist_ok=True)
            adapted = source_dir/'materialized.json';immutable_write(adapted, {'jobs':[entry]})
            wrapper = source_dir/'original_command.sh'
            body = '#!/bin/bash\n# Transport provenance; source output placeholder is intentionally unresolved.\n'+shlex.join(query['sample_argv'])+'\n'
            if wrapper.exists():native.require(wrapper.read_text()==body, 'Original command changed')
            else:wrapper.write_text(body)
            job = root/'out/vast_native_dependency'/query['id']
            if not job.exists():
                native.prepare(root, adapted, 0, query['native_runtime'], helper, wrapper,
                               root/'materialized_runs'/family/query['id'],
                               reservation.start+index, query['id'], v2_root=release,
                               v2_manifest_sha256=native.sha(release/'release_manifest.json'))
            value=native.verify(job,root)
            native.require(value['original_entry']==entry and value['fresh_start_index']==reservation.start+index,
                           'Resumed native transport differs')
            gate=check_job(root,job,kind='native_dependency');native.require(gate['allowed'],str(gate))
            record={'id':query['id'],'physical_dedup_key':physical,'owner':'/root/templates_assets',
                    'dispatch_owner':'/root','backend':'vast','workstream':'training preparation',
                    'stage':'native scene/task feasibility','class_id':query['class_id'],
                    'source':query['request'],'native_inputs_sha256':native.sha(job/'native_inputs.json'),
                    'job_dir':str(job),'observed_at':time.time(),'benchmark_credit':0,'production_released':False}
            try:
                with claim.open('x') as stream:json.dump(record,stream)
            except FileExistsError:
                skipped.append({'id':query['id'],'reason':'another backend acquired exact physical claim'});continue
            native.write(receipt,{**record,'status':'submitting'})
            jid=native.enqueue(job,root);native.bind_output(job,root)
            native.write(receipt,{**record,'status':'submitted','job_id':jid,'submitted_at':time.time()})
            submitted.append({'id':query['id'],'job_id':jid,'receipt':str(receipt),'job_dir':str(job)})
            added+=1
            if added%10==0:print(json.dumps({'prepared_queued':len(submitted),'newly_queued':added,'source_index':index}),flush=True)
        except Exception as exc:
            errors.append({'id':query['id'],'error':type(exc).__name__+': '+str(exc)})
        native.write(progress_path,{'observed_at':time.time(),'source':source_ref,'submitted':submitted,'skipped':skipped,'errors':errors})
    report={'observed_at':time.time(),'source':{'path':str(manifest),'sha256':expected_sha},
            'reservation':reservation.__dict__,'submitted':submitted,'skipped':skipped,'errors':errors,
            'newly_submitted':added,'activation_required':True,'benchmark_credit':0,'production_released':False}
    native.write(transport/'prepared.json',report)
    print(json.dumps({'submitted':len(submitted),'skipped':len(skipped),'errors':errors[:10]}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--sha256',required=True);p.add_argument('--helper',required=True)
    p.add_argument('--leave-for-cluster',type=int,required=True);p.add_argument('--transport-name',default='vast_transport')
    p.add_argument('--max-new-jobs',type=int);a=p.parse_args()
    native.require(a.leave_for_cluster>=0,'Invalid cluster reservation')
    native.require(a.max_new_jobs is None or a.max_new_jobs>0,'Invalid admission limit')
    run(a.root,a.manifest,a.sha256,a.helper,leave_for_cluster=a.leave_for_cluster,transport_name=a.transport_name,max_new_jobs=a.max_new_jobs)
