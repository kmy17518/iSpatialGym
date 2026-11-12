"""Sole suite owner for additional exact successful-member physics replays.

The canonical owner keeps its original first-member job. This owner never
recollects or resamples and never retries a failed replay automatically.
"""
from pathlib import Path
import argparse,hashlib,json,os,re,subprocess,time
from .runtime_validation_metrics import reference,require,write,validate_row
from .runtime_suite_allocations import immutable,allocation
from .production_gate import check_job
from .storage import atomic_directory_lock


def successful_members(source,row):
    from .acceptance import report_passed
    from .training_diversity import _index
    members=[];first=None
    for meta in sorted((source/'demo').glob('*/meta.json')):
        report=json.loads(meta.read_text());demo=meta.with_name('demo.hdf5');iid=meta.parent.name
        tros=list((source/'inst').rglob(iid+'-tro_state.json'))
        if report.get('success') is not True or not demo.exists() or len(tros)!=1:continue
        if first is None:first=iid
        report_passed(report,'oracle')
        require(report['instance_id']==iid,'demo member identity differs')
        idx=_index({'instance_id':iid});require(idx is not None and row['fresh_start']<=idx<row['fresh_start']+row['fresh_count'],'member outside fresh range')
        require(Path(report['provenance']['isg_root']).resolve()==Path(row['frozen_runtime']['path']).resolve(),'member demo runtime differs')
        members.append({'instance_id':iid,'demo':reference(demo),'tro':reference(tros[0]),'report':reference(meta)})
    return first,members


def reconcile_submit(entry,root,run_dir):
    comment=entry['scheduler_comment']
    commands=[['squeue','-h','-u','minyeong','-o','%i|%k'],['sacct','-X','-n','-P','-u','minyeong','-S',entry['accounting_since'],'--format=JobIDRaw,Comment%200']]
    proof=[];found=set()
    for cmd in commands:
        q=subprocess.run(cmd,capture_output=True,text=True,timeout=45,check=True);proof.append({'command':cmd,'stdout':q.stdout,'observed_at':time.time()})
        for line in q.stdout.splitlines():
            parts=line.strip().split('|')
            if len(parts)>=2 and parts[1]==comment:found.add(parts[0])
    entry['submission_reconciliation']=immutable(run_dir/'evidence/replay_submit_reconciliation',proof)
    require(len(found)<=1,'ambiguous suite replay submission')
    if found:
        entry.update(replay_job_id=found.pop(),status='replay_queued',reconciled_at=time.time())
    # Zero matches are not proof an interrupted sbatch never allocated.


def cycle(root,run_dir,helper):
    from .dispatch_cluster_first import probe
    from .replay_scheduling import dual_partition_resources
    from .replay_migration import migrate
    errors=[];submitted=[];observed=None;resources=None
    for rowpath in sorted((run_dir/'rows').glob('*.json')):
        try:
            row=json.loads(rowpath.read_text());validate_row(row)
            if row['camera_exempt']:continue
            rp=root/'dispatch/receipts'/(row['dispatch_token']+'.json')
            if not rp.exists():continue
            receipt=json.loads(rp.read_text())
            if receipt.get('status')!='submitted':continue
            a=allocation(row,receipt,root);source=Path(a['job_dir'])
            if not (source/'sampling.done').exists():continue
            first,members=successful_members(source,row)
            if not members:continue
            target=run_dir/'member_replays'/rowpath.name;state=json.loads(target.read_text()) if target.exists() else {'row':reference(rowpath),'source_allocation':a,'members':{}}
            require(state['row']==reference(rowpath) and state['source_allocation']==a,'suite replay source ownership changed')
            state['canonical_first_member']=first
            for member in members:
                iid=member['instance_id']
                if iid==first:continue
                key=str(a['jobid'])+'_'+iid;entry=state['members'].get(iid)
                save=lambda:write(target,state)
                if entry:
                    require(entry['inputs']==member,'owned replay inputs changed')
                    if entry['status']=='submitting':reconcile_submit(entry,root,run_dir);save()
                    gate=Path(entry['output'])/'gate.json'
                    if gate.exists():entry.update(status='replay_terminal',gate=reference(gate));save();continue
                    if entry.get('replay_job_id') and entry.get('replay_backend')!='vast':
                        try:migrate(entry,root,row['frozen_runtime']['path'],helper,0,save)
                        except Exception as exc:entry['pending_disposition']={'observed_at':time.time(),'error':repr(exc),'owner':'compute_runtime_suite'};save()
                    continue
                decision=check_job(root,source,kind='source_validation');require(decision['allowed'],'source authorization: '+decision['reason'])
                if observed is None:
                    observed=probe();require(observed['state']=='available','fresh both-account scheduler probe unavailable')
                    raw='\n'.join(' '.join(k+'='+str(v) for k,v in n.items()) for n in observed['nodes']);resources,policy=dual_partition_resources(raw)
                    observed['replay_route']=policy;observed['secondary_route_blocker']='Secondary writable replay caches/private source closure not yet proved; primary broad compatible route plus exact typed Vast overflow owned here.'
                require(time.time()-observed['observed_at']<120,'scheduler observation expired')
                output=root/'validation/runtime_suites'/row['run_id']/row['class_id']/iid
                require(not output.exists(),'fresh exact member output required')
                comment='campaign_20261110_replay_'+key
                command=['sbatch','--parsable',*resources,'-c','8','--mem=32G','--time=00:40:00','-J','isg261110_replay','--comment',comment,
                    '--export=ALL,ISGF_REPLAY_SCENE_HELPER='+str(helper/'scripts/campaign_20261110/infra_replay_scene.py')+',ISGF_REPLAY_GATE_HELPER='+str(helper/'scripts/campaign_20261110/replay_gate_quality.py'),
                    '-o',str(root/'slurm/%x_%j.log'),str(helper/'scripts/campaign_20261110/infra_open_loop.sh'),str(source),row['frozen_runtime']['path'],str(output),iid]
                entry={'status':'submitting','source_job_id':key,'source_allocation_id':str(a['jobid']),'source_job_dir':str(source),'instance_id':iid,'inputs':member,'output':str(output),
                    'scheduler_comment':comment,'accounting_since':time.strftime('%Y-%m-%d',time.gmtime()),'command':command,'capacity':immutable(run_dir/'evidence/replay_capacity',observed),
                    'source_gate':decision,'wrapper':reference(helper/'scripts/campaign_20261110/infra_open_loop.sh'),'helper_snapshot':str(helper),'created_at':time.time()}
                state['members'][iid]=entry;save()
                q=subprocess.run(command,capture_output=True,text=True,timeout=60)
                entry['submission']={'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr,'observed_at':time.time()}
                jid=q.stdout.strip().split(';')[0]
                if q.returncode==0 and jid.isdigit():entry.update(status='replay_queued',replay_job_id=jid,replay_backend='cluster');submitted.append(jid)
                else:entry['status']='submission_unresolved'
                save()
        except Exception as exc:errors.append({'class_id':rowpath.stem,'error':repr(exc)})
    result={'observed_at':time.time(),'submitted':submitted,'errors':errors,'owner':'compute_runtime_suite','scope':'Additional exact successful members only; canonical first-member owner preserved; zero-age exact Vast migration.'}
    write(run_dir/'member_replay_observer.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--helper',type=Path,required=True);p.add_argument('--every',type=float,default=20);a=p.parse_args()
    with atomic_directory_lock(a.run_dir/'member_replay_observer.lockdir'):
        while True:
            try:print(json.dumps(cycle(a.root,a.run_dir,a.helper)),flush=True)
            except Exception as e:print(json.dumps({'error':repr(e),'observed_at':time.time()}),flush=True)
            if not a.every:return
            time.sleep(a.every)
if __name__=='__main__':main()
