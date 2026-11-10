"""Launch one immutable camera proposal with strict input and output provenance."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(entry,*,root,manifest_root,inputs_root,output_root,index):
    root=Path(root).resolve();inputs_root=Path(inputs_root).resolve();output_root=Path(output_root).resolve()
    require(inputs_root.is_relative_to(root) and output_root.is_relative_to(root),'native paths outside campaign')
    require(entry.get('task','').startswith('camera_relocalization-'),'native launcher only accepts camera proposals')
    require(not entry.get('blockers',[]) and not entry.get('required_new_grasp_key'),'camera proposal is blocked')
    argv=[str(item).replace('${MATERIALIZED}',str(inputs_root)).replace('${ROOM_MATERIALIZED}',str(inputs_root))
          .replace('${CAMPAIGN_OUT}',str(output_root)) for item in entry['sample_argv']]
    require(argv[:3]==['python','-m','ispatialgym.sampling.sample_isg'],'unexpected native command')
    require('--no-write-back' in argv and '--start-index' not in argv,'native proposal can mutate historical state or reuse indices')
    def option(flag):
        require(argv.count(flag)==1,'missing or duplicate native option '+flag)
        return argv[argv.index(flag)+1]
    require(option('--variant')==entry['alias'] and option('--scene')==entry['scene'] and option('--room')==entry['room'],
            'native command differs from proposal identity')
    require(option('--templates')=='1' and option('--instances-per-template')=='1','bootstrap must create one fresh source')
    refs={}
    for flag,is_directory in (('--customs',False),('--bddl-root',True)):
        path=Path(option(flag)).resolve();require(path.is_relative_to(root),'native definition outside campaign')
        require(path.is_dir() if is_directory else path.is_file(),'native input absent: '+str(path))
        if is_directory:
            definitions=list(path.rglob('*.bddl'));require(bool(definitions),'empty native definition root')
            refs.update({str(item):sha(item) for item in definitions})
        else:refs[str(path)]=sha(path)
    output=Path(option('--out')).resolve();require(output.is_relative_to(output_root),'native output outside fresh run')
    require(not output.exists(),'native output already exists; reserve fresh indices/output')
    environment=entry.get('environment',{})
    require(set(environment)<= {'ISPATIALGYM_CAMPAIGN_OBSTACLE_GOAL'},'unexpected camera proposal environment')
    require(all(value=='1' for value in environment.values()),'invalid stationary obstacle flag')
    if entry['task'].endswith('obstructed'):
        require(environment.get('ISPATIALGYM_CAMPAIGN_OBSTACLE_GOAL')=='1','obstacle proposal must preserve stationary goal')
    argv+=['--start-index',str(index),'--time-budget-s','1800']
    return argv,output,environment,refs


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('entry',type=int);args=p.parse_args()
    root=Path(os.environ['CROOT']).resolve();materialized=Path(os.environ['MATERIALIZED']).resolve()
    manifest=materialized/'validation_jobs.json';entry=json.loads(manifest.read_text())['jobs'][args.entry]
    output=root/'materialized_runs'/os.environ['MATERIALIZED_RUN_TAG'];index=int(os.environ['MATERIALIZED_BASE_INDEX'])+args.entry
    argv,job,extra,refs=prepare(entry,root=root,manifest_root=materialized,
        inputs_root=os.environ.get('CAMERA_INPUT_ROOT',str(materialized)),output_root=output,index=index)
    job.mkdir(parents=True,exist_ok=False)
    record={'argv':argv,'manifest':str(manifest),'manifest_sha256':sha(manifest),'entry':args.entry,
            'source_snapshot':os.environ['ISGE_CODE'],'inputs':refs,'environment':extra,
            'slurm_job_id':os.environ['SLURM_JOB_ID'],'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (job/'command.json').write_text(json.dumps(record,indent=2)+'\n')
    with (job/'sample.log').open('w') as log:
        try:rc=subprocess.run(argv,env={**os.environ,**extra},stdout=log,stderr=subprocess.STDOUT,timeout=2400).returncode
        except subprocess.TimeoutExpired:rc=124
    tros=list(job.rglob('*-tro_state.json'))
    result={'returncode':rc,'tro_count':len(tros),'tros':[str(path) for path in tros],
            'class_id':entry['class_id'],'fresh_start_index':index,'validation_only':True,
            'finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (job/'sampling_gate.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result));raise SystemExit(0 if rc==0 and len(tros)==1 else 2)


if __name__=='__main__':main()
