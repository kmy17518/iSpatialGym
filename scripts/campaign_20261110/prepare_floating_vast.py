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
    argv[argv.index('--out')+1] = '${CAMPAIGN_OUT}/floating_feasibility/'+query['id']
    return {'class_id':query['class_id'], 'scene':query['scene'], 'environment':{},
            'sample_argv':argv, 'production_released':False,
            'source_query_sha256':query['request']['sha256'],
            'source_query_id':query['id'], 'physical_dedup_key':query['physical_dedup_key']}


def immutable_write(path, value):
    if path.exists():
        native.require(json.loads(path.read_text()) == value, 'Prepared transport changed: '+str(path))
    else:
        native.write(path, value)


def run(root, manifest, expected_sha, helper, *, leave_for_cluster):
    root, manifest, helper = map(lambda p:Path(p).resolve(), (root, manifest, helper))
    native.require(native.sha(manifest) == expected_sha, 'Source manifest changed')
    source = json.loads(manifest.read_text())
    native.require(source['campaign_id'] == root.name and source['benchmark_credit'] == 0
                   and source['production_authorized'] is False, 'Source purpose differs')
    directory = root/'training_preparation/model_variant_inventory_r2/floating_feasibility_execution_r1'
    transport = directory/'vast_transport';transport.mkdir(parents=True, exist_ok=True)
    for name in ('claims', 'receipts'):
        (directory/name).mkdir(parents=True, exist_ok=True)
    queries = source['jobs']
    reservation = reserve(root, 'floating_feasibility_vast_r1_'+expected_sha[:16], len(queries))
    digest_cache = {}
    def checked(ref):
        name = ref['path']
        if name not in digest_cache:digest_cache[name] = native.sha(name)
        native.require(digest_cache[name] == ref['sha256'], 'Source input changed: '+name)
    releases = {};submitted = [];skipped = [];errors = []
    for index, query in enumerate(queries):
        physical = query['physical_dedup_key']
        claim = directory/'claims'/(physical+'.json');receipt = directory/'receipts'/(physical+'.json')
        if index < leave_for_cluster or claim.exists() or receipt.exists():
            skipped.append({'id':query['id'], 'reason':'cluster prefix or existing physical ownership'});continue
        try:
            entry = normalized_entry(query)
            for ref in query['input_refs']+[query['request']]:checked(ref)
            original_release = Path(query['dependency_release'])
            if str(original_release) not in releases:
                release = transport/'releases'/original_release.name
                if not release.exists():shutil.copytree(original_release, release)
                files = {}
                for path in original_release.rglob('*'):
                    if path.is_file():
                        relative = str(path.relative_to(original_release));digest=native.sha(path)
                        native.require(native.sha(release/relative)==digest, 'Library copy differs')
                        files[relative]=digest
                native.require('anchors.json' in files and any(n.endswith('/library.json') for n in files),
                               'Incomplete exact library release')
                immutable_write(release/'release_manifest.json', {'files':files, 'source':query['request']})
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
                               root/'materialized_runs/floating_feasibility'/query['id'],
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
            if len(submitted)%10==0:print(json.dumps({'prepared_queued':len(submitted),'source_index':index}),flush=True)
        except Exception as exc:
            errors.append({'id':query['id'],'error':type(exc).__name__+': '+str(exc)})
        native.write(transport/'progress.json',{'observed_at':time.time(),'submitted':submitted,'skipped':skipped,'errors':errors})
    report={'observed_at':time.time(),'source':{'path':str(manifest),'sha256':expected_sha},
            'reservation':reservation.__dict__,'submitted':submitted,'skipped':skipped,'errors':errors,
            'activation_required':True,'benchmark_credit':0,'production_released':False}
    native.write(transport/'prepared.json',report)
    print(json.dumps({'submitted':len(submitted),'skipped':len(skipped),'errors':errors[:10]}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--sha256',required=True);p.add_argument('--helper',required=True)
    p.add_argument('--leave-for-cluster',type=int,required=True);a=p.parse_args()
    native.require(a.leave_for_cluster>=0,'Invalid cluster reservation')
    run(a.root,a.manifest,a.sha256,a.helper,leave_for_cluster=a.leave_for_cluster)
