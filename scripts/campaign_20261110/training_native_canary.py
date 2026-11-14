"""Run one explicitly authorized native training canary; never accept or refill."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from scripts.campaign_20261110.contracts import file_digest, require, write_json_atomic
from scripts.campaign_20261110.production_gate import check_environment


def prepare(root, requests, index):
    root, requests = Path(root).resolve(), Path(requests).resolve()
    require(requests.is_relative_to(root), 'canary request outside campaign')
    document = json.loads(requests.read_text())
    require(document['campaign_id']=='campaign_20261110' and document['purpose']=='training_pipeline_canary', 'wrong canary manifest')
    row = document['records'][index]
    require(row['count']==row['max_instances']==1 and row['production_released'] is False, 'one-instance bounded canary required')
    for field, sha_field in [('config_path','config_sha256'),('authorization_path','authorization_sha256'),('input_manifest','input_manifest_sha256')]:
        path=Path(row[field]).resolve()
        require(path.is_relative_to(root) and file_digest(path)==row[sha_field], 'canary input identity changed: '+field)
    config=json.loads(Path(row['config_path']).read_text())
    authorization=json.loads(Path(row['authorization_path']).read_text())
    from .training_canary_scope import validate_scope
    validate_scope(config, authorization)
    require(file_digest(Path(row['native_runtime'])/'.snapshot_commit')==config['validation_key']['code_sha256']==authorization['native_runtime_sha256'], 'native runtime changed')
    job=json.loads(Path(row['input_manifest']).read_text())['jobs'][row['manifest_index']]
    require(job['candidate_id']==row['candidate_id'] and job['class_id']==row['class_id'], 'candidate identity changed')
    require(config['perturbations']==job['training_scale_plan'], 'materialized physical scale plan differs')
    environment={**row['environment'],'ISGF_ROOT':str(root)}
    require(environment=={**job['environment'],'ISPATIALGYM_CAMPAIGN_CONFIG':row['config_path'],'ISGF_ROOT':str(root)}, 'native candidate environment changed')
    release=Path(job['dependency_release']).resolve();manifest=release/'release_manifest.json'
    require(release.is_relative_to(root) and file_digest(manifest)==authorization['dependency_release_sha256'], 'exact dependency release changed')
    for relative,expected in json.loads(manifest.read_text())['files'].items():
        path=(release/relative).resolve()
        require(path.is_relative_to(release) and file_digest(path)==expected, 'dependency payload changed: '+relative)
    require(environment['ISPATIALGYM_CAMPAIGN_CONFIG']==row['config_path'], 'environment configuration differs')
    decision=check_environment(root,environment,kind='collection',start=row['start'],count=row['count'])
    require(decision['allowed'] and decision['category']=='validation_canary', 'production gate: '+decision['reason'])
    argv=list(job['sample_argv'])
    require(argv[:3]==['python','-m','ispatialgym.sampling.sample_isg'], 'unsupported native entry point')
    require('--no-write-back' in argv and '--start-index' not in argv and '--time-budget-s' not in argv, 'native entry point lacks bounds')
    require(argv[argv.index('--templates')+1]=='1' and argv[argv.index('--instances-per-template')+1]=='1' and argv[argv.index('--max-attempts')+1]=='3', 'native draw bound changed')
    require(authorization['max_attempts']==3 and authorization['sample_time_budget_seconds']==1800, 'authorization draw/time bound differs')
    output=Path(row['output']).resolve()
    require(output.is_relative_to(Path(row['config_path']).parent) and not output.exists(), 'new private canary output required')
    argv[0]=sys.executable
    argv[argv.index('--out')+1]=str(output)
    argv+=['--start-index',str(row['start']),'--time-budget-s','1800']
    return row,job,environment,decision,argv,output


def verify_native_rooms(problem, job, config):
    """Keep the exact candidate's complete room set, including multi-room doors."""
    template = config['template']
    expected = template.get('scene_setup', {}).get('task_rooms', [template['room']])
    actual = problem.rooms(job['scene'])
    require(template['scene'] == job['scene'] and template['room'] == job['room']
            and isinstance(expected, list) and expected and len(expected) == len(set(expected))
            and job['room'] in expected and len(actual) == len(set(actual))
            and set(actual) == set(expected), 'native room binding differs from the exact candidate')
    return list(actual)


def run(root, requests, index, *, preflight=False):
    row,job,environment,decision,argv,output=prepare(root,requests,index)
    # Parsing the real native declaration catches role/name/BDDL errors before
    # allocating GPU simulation. No synthetic fixture can substitute for this.
    os.environ.update(environment)
    from ispatialgym.sampling.v2.problem import GroundedProblem
    from ispatialgym.sampling import campaign_runtime
    gp=GroundedProblem.load(job['alias'],argv[argv.index('--customs')+1],argv[argv.index('--bddl-root')+1])
    from types import SimpleNamespace
    from ispatialgym.sampling.sample_isg import variant_split
    args=SimpleNamespace(seed=int(argv[argv.index('--seed')+1]),seed_index=0,seed_split=None,
        scene=job['scene'],room=argv[argv.index('--room')+1] if '--room' in argv else None,scene_from_split=False)
    variant_split(args,gp)
    require(file_digest(row['config_path']) == row['config_sha256'], 'native configuration changed during declaration parsing')
    native_rooms = verify_native_rooms(gp, job, json.loads(Path(row['config_path']).read_text()))
    campaign_runtime.configuration()
    proof={'campaign_id':'campaign_20261110','candidate_id':row['candidate_id'],'request_sha256':file_digest(requests),
           'config_sha256':row['config_sha256'],'gate':decision,'argv':argv,'native_runtime':row['native_runtime'],
           'created_at':dt.datetime.now(dt.timezone.utc).isoformat(),'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
           'cpu_affinity':sorted(os.sched_getaffinity(0)),'native_task_rooms':native_rooms,
           'production_released':False,'eligible_accepted':0}
    if preflight:
        proof['native_declaration_parsed']=True
        write_json_atomic(Path(row['config_path']).parent/'preflight.json',proof)
        print(json.dumps({'candidate_id':row['candidate_id'],'preflight_passed':True}));return
    output.mkdir(parents=True)
    write_json_atomic(output/'command.json',proof)
    start=time.time()
    with (output/'sample.log').open('w') as log:
        try:rc=subprocess.run(argv,env={**os.environ,**environment},stdout=log,stderr=subprocess.STDOUT,timeout=2400).returncode
        except subprocess.TimeoutExpired:rc=124
    tros=list(output.rglob('*-tro_state.json'))
    result={**proof,'returncode':rc,'tro_count':len(tros),'tros':[{'path':str(p),'sha256':file_digest(p)}for p in tros],
            'elapsed_seconds':time.time()-start,'finished_at':dt.datetime.now(dt.timezone.utc).isoformat(),
            'next_required':['native binding','successful demo','own native replay','hard QA','group compatibility','augmentation and batched restoration','full pipeline distribution validation']}
    write_json_atomic(output/'sampling_gate.json',result)
    print(json.dumps({'candidate_id':row['candidate_id'],'returncode':rc,'tro_count':len(tros),'eligible_accepted':0}))
    require(rc==0 and len(tros)==1, 'bounded native canary failed; inspect preserved evidence before retry')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--requests',type=Path,required=True);p.add_argument('--index',type=int,required=True);p.add_argument('--preflight',action='store_true')
    a=p.parse_args();run(a.root,a.requests,a.index,preflight=a.preflight)
