"""Read-only projection of the eleven exact sequential restoration controls."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from .contracts import file_digest, require, write_json_atomic

PHYSICS=('fresh_A','fresh_B','successful_AA','failed_AA','interrupted_AA','ABA')
RENDER=('fresh_A','fresh_B','ABA')

def read(path):return json.loads(Path(path).read_text())
def ref(path):return {'path':str(Path(path).resolve()),'sha256':file_digest(path)}
def checked(root,reference):
    p=Path(reference['path']).resolve()
    require(p.is_relative_to(Path(root).resolve())and file_digest(p)==reference['sha256'],'control evidence changed')
    return read(p)

def control(root,batch,stage,name,scheduler,observation,*,directory=None):
    cleanup=stage=='renderer_cleanup'
    directory=Path(directory)if directory else Path(batch)/('renderer_cleanup_r1'if cleanup else'sequential_verification_r1')
    d=directory/name if cleanup else directory/stage/name
    out={'stage':stage,'control':name,'status':'awaiting_request','scheduler_state':None,'job_id':None,
         'verification_status':'not_verified','evidence':{},'next_action':'Existing author prepares exact bounded request'}
    try:
        request=d/'request.json'
        if not request.exists():return out
        request_ref=ref(request);checked(root,request_ref);out['evidence']['request']=request_ref
        out.update(status='awaiting_preflight',next_action='Existing author verifies exact source and runtime bindings')
        preflight=d/'preflight.json'
        if preflight.exists():out['evidence']['preflight']=ref(preflight);out.update(status='awaiting_dispatch',next_action='Sole compute callback dispatches this exact control')
        submission=d/'submission.json'
        if submission.exists():
            value=read(submission);require(value['request']==request_ref,'submission request differs')
            rows=value.get('jobs');require(isinstance(rows,list)and len(rows)==1,'ambiguous control allocation')
            job=rows[0];require(job['request']==request_ref and job['returncode']==0 and job['control']==name,'control submission binding differs')
            out.update(job_id=str(job['job_id']),status='awaiting_scheduler_observation',next_action='Read actual scheduler state; submission alone is not execution')
            out['evidence']['submission']=ref(submission)
            actual=scheduler.get(str(job['job_id']))
            if actual:
                out.update(scheduler_state=actual['state'],scheduler_observed_at=observation['observed_at'],scheduler=actual)
                out['evidence']['scheduler']=observation['source']
                if actual['state']in('RUNNING','COMPLETING'):
                    out.update(status='running',next_action='Preserve current allocation; inspect native output as it arrives')
                elif actual['state']in('PENDING','CONFIGURING'):
                    out.update(status='queued',next_action='Sole compute owner reconciles exact pending allocation')
                elif actual['state']=='COMPLETED'and actual['exit_code']=='0:0':
                    out.update(status='awaiting_strict_verification',next_action='Strict owner checks native state, source identities and applicable cleanup evidence')
                else:out.update(status='execution_failed',next_action='Diagnose exact terminal log; preserve inputs and do not automatically retry')
        probe=d/'output/probe.json'
        if probe.exists():
            probe_ref=ref(probe);value=checked(root,probe_ref);out['evidence']['probe']=probe_ref
            out['probe_observed_at']=value.get('observed_at')
            out['world_construction_unreviewed']=value.get('world_construction')
            out['episodes_observed']=len(value.get('entries',[]))
        execution=d/'execution_observation.json'
        if execution.exists():out['evidence']['execution']=ref(execution)
        verification=directory/'verification.json'
        rendering_stage=directory/'rendering_verification.json'
        if stage=='rendering'and not verification.exists()and rendering_stage.exists():
            verification=rendering_stage
            scoped=read(verification)
            require(scoped.get('schema')=='isg-sequential-rendering-verification-v1'and
                    scoped.get('verified_stages')==['rendering']and scoped.get('full_matrix_verified')is False,
                    'rendering-only verification scope differs')
        if verification.exists():
            value=read(verification)
            entry=value.get('controls',{}).get(name)if cleanup else value.get('stages',{}).get(stage,{}).get(name)
            require(value.get('status')=='verified'and entry and entry['probe']==out['evidence'].get('probe'),
                    'strict verification does not bind current probe')
            require(out['scheduler_state']=='COMPLETED'and out['scheduler']['exit_code']=='0:0','strict proof lacks current clean terminal observation')
            out['evidence']['verification']=ref(verification)
            out.update(status='verified',verification_status='verified',next_action='Retain exact verified artifacts; no new execution')
    except Exception as error:out.update(status='evidence_error',error=f'{type(error).__name__}: {error}',next_action='Repair exact evidence binding; never infer a verification pass')
    return out

def project(root,batch,scheduler,observation):
    rows=[control(root,batch,stage,name,scheduler,observation)for stage,names in
          [('physics',PHYSICS),('rendering',RENDER),('renderer_cleanup',('failure','interruption'))]for name in names]
    state_path=Path(batch)/'stream/state.json';state=read(state_path)if state_path.exists()else{}
    prospective=[];manifest=Path(batch)/'sequential_anchor_r2/requests.json'
    if manifest.exists():
        value=read(manifest)
        require(len(value['records'])==6 and {r['control']for r in value['records']}==set(PHYSICS)and
                all(r['stage']=='physics'for r in value['records']),'prospective matrix scope differs')
        for row in value['records']:
            require(row['request']==ref(manifest.parent/'physics'/row['control']/'request.json'),
                    'prospective request binding differs')
            prospective.append(control(root,batch,'physics',row['control'],scheduler,observation,directory=manifest.parent))
    return {'schema':'isg-sequential-control-status-v1','campaign_id':'campaign_20261110',
        'observed_at':datetime.now(timezone.utc).isoformat(),'owner':'/root/dataset_contract',
        'strict_source_observed_at':state.get('observed_at'),'strict_state':ref(state_path)if state_path.exists()else None,
        'strict_errors':{k:state[k]for k in ('matrix_error','verification_error','renderer_cleanup_error','cycle_error')if state.get(k)},
        'scheduler_observed_at':observation['observed_at'],'controls':rows,'required_controls':11,
        'prospective_physics_manifest':ref(manifest)if manifest.exists()else None,
        'prospective_controls':prospective,'prospective_verified_controls':sum(r['verification_status']=='verified'for r in prospective),
        'verified_controls':sum(r['verification_status']=='verified'for r in rows),'production_released':False,'benchmark_credit':0}

def observe(root,batch):
    jobs=[]
    for directory in (Path(batch)/'sequential_verification_r1',Path(batch)/'renderer_cleanup_r1',Path(batch)/'sequential_anchor_r2'):
        for p in directory.rglob('submission.json'):
            for j in read(p).get('jobs',[]):
                if j.get('job_id'):jobs.append(str(j['job_id']))
    now=datetime.now(timezone.utc).isoformat();directory=Path(batch)/'control_status_evidence';directory.mkdir(exist_ok=True)
    command=['sacct','-n','-P','-j',','.join(sorted(set(jobs))),'--format=JobIDRaw,State,ExitCode,Start,End,ElapsedRaw,NodeList']
    result=subprocess.run(command,text=True,capture_output=True,check=True,env=os.environ|{'TZ':'UTC'})if jobs else None
    raw=result.stdout if result else '';p=directory/('scheduler_'+str(time.time_ns())+'.txt');p.write_text(raw);p.chmod(0o640)
    scheduler={}
    for line in raw.splitlines():
        fields=line.split('|')
        if len(fields)>=7 and fields[0]in jobs:
            scheduler[fields[0]]=dict(zip(('job_id','state','exit_code','started_at','ended_at','elapsed_seconds','node'),fields))
    return project(root,batch,scheduler,{'observed_at':now,'source':ref(p)})

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--batch',type=Path,required=True);p.add_argument('--interval',type=float,default=30);p.add_argument('--once',action='store_true');a=p.parse_args()
    with(a.batch/'control_status.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            value=observe(a.root,a.batch);path=a.batch/'control_status.json';write_json_atomic(path,value);path.chmod(0o640)
            print(json.dumps({'observed_at':value['observed_at'],'controls':len(value['controls']),'verified':value['verified_controls']}),flush=True)
            if a.once:break
            time.sleep(a.interval)
if __name__=='__main__':main()
