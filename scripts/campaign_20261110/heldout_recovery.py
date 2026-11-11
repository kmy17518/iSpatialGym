"""Read-only exact held-out quota and controller audit; no collection side effects."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from contextlib import ExitStack
from datetime import datetime,timezone
import json,pathlib,shutil,sqlite3,tempfile,time,subprocess
from scripts.campaign_20261110.contracts import task_quotas,require,file_digest,write_json_atomic
from scripts.campaign_20261110.current_eligibility import assess,read_ledger
from scripts.campaign_20261110.dashboard_evidence import standard_catalog,observed_json,failure_observation
from scripts.campaign_20261110.export_release import CAMERA_STATES
from scripts.campaign_20261110.storage import atomic_directory_lock

SPLITS=('eval_model','eval_category')
def iso(t):return datetime.fromtimestamp(t,timezone.utc).isoformat() if t else None
def ref(p):return {'path':str(p),'sha256':file_digest(p)}
def read(p):return json.loads(pathlib.Path(p).read_text())
def recovery_state(remaining,jobs,latest):
    if not remaining:return 'complete'
    states={j.get('worker_status')for j in jobs}
    if states & {'assigned','running','RUNNING','COMPLETING'}:return 'running'
    if states & {'queued','staging','PENDING','CONFIGURING'}:return 'queued'
    if jobs:return 'awaiting_observation'
    status=(latest or{}).get('status')
    if status in ('native_preparation','engineering_blocked'):return 'engineering_blocked'
    if status=='native_passed':return 'awaiting_validation'
    if status in ('admitted','complete','retired_after_claim'):return 'ready_for_scoped_refill'
    if status in ('external_submission','awaiting_replay','awaiting_gate'):return 'awaiting_acceptance'
    if status=='failed' or (latest or{}).get('reason'):return 'failed'
    return 'unattempted'

def scheduler_states(ids):
    """Read exact Slurm states; submission receipts never imply pending GPUs."""
    if not ids:return {},[]
    states={};errors=[];joined=','.join(sorted(ids))
    commands=[['squeue','-h','-j',joined,'-o','%i|%T|%R'],
              ['sacct','-n','-P','-j',joined,'-o','JobIDRaw,State,Reason,ExitCode']]
    for command in commands:
        try:
            p=subprocess.run(command,capture_output=True,text=True,timeout=20,check=True)
            for line in p.stdout.splitlines():
                parts=line.strip().split('|')
                if len(parts)>=3 and parts[0]in ids:
                    states[parts[0]]={'worker_status':parts[1].split()[0],
                        'pending_reason':parts[2]if parts[1]=='PENDING'else None,
                        'scheduler_observed_at':iso(time.time()),'scheduler_source':command[0]}
        except (OSError,subprocess.SubprocessError)as exc:errors.append({'source':command[0],'error':str(exc)[:500]})
    return states,errors
def quota_bucket(task,value,has_buckets):
    if not has_buckets:return None
    if task.startswith('camera_relocalization-'):
        require(str(value) in ('3','5','10'),'unknown native camera radius bucket')
        return ('3','5','10').index(str(value))
    require(isinstance(value,int) and not isinstance(value,bool),'unknown native evaluation bucket')
    return value
def targets(catalog):
    quotas={q.task:q for q in task_quotas()};require(len(catalog)==100 and Counter(catalog.values())==Counter({q.task:q.standard_classes for q in quotas.values()}),'incomplete canonical inventory')
    rows=[]
    for cls,task in sorted(catalog.items()):
        q=quotas[task]
        if not q.model_classes:continue
        buckets=list(enumerate(q.model_buckets)) if q.model_buckets else [(None,q.model_instances_per_class)]
        rows.extend({'split':'eval_model','class_id':cls,'task':task,'bucket':b,'target':n}for b,n in buckets)
    for task,q in sorted(quotas.items()):
        buckets=list(enumerate(q.category_buckets)) if q.category_buckets else [(None,q.category_instances)]
        rows.extend({'split':'eval_category','class_id':task,'task':task,'bucket':b,'target':n}for b,n in buckets)
    require(sum(r['target']for r in rows if r['split']=='eval_model')==891,'model quota changed')
    require(sum(r['target']for r in rows if r['split']=='eval_category')==109,'category quota changed')
    return rows

def snapshot(root):
    db=root/'dataset/allocations.sqlite';sources=[(db,db.with_suffix('.lockdir'))]+[(root/'dataset'/n,root/'dataset'/(n+'.lockdir'))for n in CAMERA_STATES];rows=[];groups={};proof=[]
    with ExitStack()as locks,tempfile.TemporaryDirectory(prefix='held-recovery-')as td:
        for _,lock in sorted(sources):locks.enter_context(atomic_directory_lock(lock))
        for p,_ in sources:
            if not p.exists():continue
            local=pathlib.Path(td)/p.name;shutil.copyfile(p,local);proof.append({'path':str(p),'sha256':file_digest(local)})
            if p==db:
                with sqlite3.connect(f'file:{local}?mode=ro',uri=True)as con:
                    require(con.execute('PRAGMA quick_check').fetchall()==[('ok',)],'allocation DB integrity failed')
                    for iid,gid,split,raw in con.execute("SELECT id,group_id,split,record FROM members WHERE split IN ('eval_model','eval_category')"):
                        r=json.loads(raw);require(r['instance_id']==iid and r['group_id']==gid and r['split']==split,'allocation identity differs');rows.append(r)
                        if gid not in groups:groups[gid]=json.loads(con.execute('SELECT record FROM groups WHERE id=?',(gid,)).fetchone()[0])
                        require(groups[gid]['accepted'].get(iid)==split and r in groups[gid]['members'],'uncommitted held allocation')
            else:
                d=read(local)
                for iid,r in d.get('accepted',{}).items():
                    if r['split']in SPLITS:require(iid==r['instance_id'],'camera identity mismatch');rows.append({**r,'task':d['task']})
    require(len({r['instance_id']for r in rows})==len(rows),'duplicate held identity')
    return rows,groups,proof

def publish(root):
    root=pathlib.Path(root).resolve();D=root/'infra/heldout_recovery';D.mkdir(parents=True,exist_ok=True);now=time.time();errors=[];sources=[];cache={}
    def load(p,kind='input'):
        p=pathlib.Path(p).resolve();require(p.is_relative_to(root),'held audit source outside campaign')
        if str(p)not in cache:
            value,source=observed_json(p,kind);cache[str(p)]=value;sources.append(source)
        return cache[str(p)]
    catalogpath=root/'manifests/dashboard_standard_catalog.json';catalog=standard_catalog(catalogpath);sources.append(ref(catalogpath));rows=targets(catalog)
    accepted,groups,snapshot_sources=snapshot(root);ledger,lref=read_ledger(root);eligible,_=assess(root,accepted,groups,ledger);counts=Counter();rawcounts=Counter();last=defaultdict(float)
    quotas={q.task:q for q in task_quotas()}
    for r in accepted:
        task=r['task'];split=r['split'];q=quotas[task];bucket=quota_bucket(task,r.get('bucket'),getattr(q,'model_buckets'if split=='eval_model'else'category_buckets'))
        cls=r.get('class_id')or r.get('contract',{}).get('class_id')or task;scope=cls if split=='eval_model'else task;key=(split,scope,bucket);rawcounts[key]+=1
        if eligible[r['instance_id']]['quota_eligible']:counts[key]+=1
        p=root/'dataset/groups'/(r.get('group_id','')+'.json')
        if p.is_file():last[key]=max(last[key],p.stat().st_mtime)
    # The active supervisor manifest selects the precise configs during helper handoff.
    manifest=root/'infra/handoffs/eval100_impact9a/manifest.json';workers=list(load(manifest)['workers']);observations=defaultdict(list);ownership=defaultdict(list)
    # Fresh, narrowly owned recovery lanes may be handed off separately from
    # the main supervisor. Read their exact prepared manifest without treating
    # preparation or a queue receipt as successful validation.
    recovery=root/'infra/heldout_recovery/prepared_source_recovery_r2.json'
    if recovery.exists():
        known={str(pathlib.Path(w['config']).resolve())for w in workers}
        for directory in sorted({r['controller']for r in load(recovery)['records']}):
            directory=pathlib.Path(directory);config=directory/'config.json'
            if str(config.resolve())not in known:
                workers.append({'name':directory.name,'config':str(config),'state':str(directory/'state.json')})
                known.add(str(config.resolve()))
    for filename in ('prepared_category_jar_upper_r1.json','prepared_scoped_refill_r1.json',
                     'prepared_camera_hamper_recovery_r1.json','prepared_scaling_model_admission_r1.json'):
        receipt=root/'infra/heldout_recovery'/filename
        if receipt.exists():
            directory=pathlib.Path(load(receipt)['controller'])
            workers.append({'name':directory.name,'config':str(directory/'config.json'),'state':str(directory/'state.json')})
    pipeline=load(root/'manifests/pipeline_status.json');prepared=defaultdict(list)
    for r in pipeline['rows']:
        if r['split']in SPLITS:prepared[(r['split'],r['class_id']if r['split']=='eval_model'else r['task'])].append(r)
    def scope_of(path):
        cfg=load(path,'campaign_config');t=cfg['template'];return cfg,t,(cfg['split'],t['class_id']if cfg['split']=='eval_model'else t['task'])
    def observation(record,worker,cfg,state,seed=None):
        path=(record.get('prepared')or{}).get('config')or(seed or{}).get('seed_config')
        command=record.get('command',[])
        if not path:path=next((s.split('=',1)[1]for s in command if isinstance(s,str)and s.startswith('ISPATIALGYM_CAMPAIGN_CONFIG=')),None)
        if not path and(seed or{}).get('inputs'):path=str(pathlib.Path(seed['inputs'])/'campaign_config.json')
        if not path:return
        cc,t,key=scope_of(path)
        if key[0]not in SPLITS:return
        a=record.get('allocation');status=record.get('status',state.get('status','unobserved'));job=None
        receipt=record.get('dispatch_receipt')
        if not a and receipt and pathlib.Path(receipt).exists():
            routed=load(receipt,'dispatch_receipt');a=routed.get('allocation')
            if a:status=routed.get('status',status)
        if a:
            job=dict(job_id=a['jobid'],job_dir=a['job_dir'],runtime=a['code'],controller_status=status)
            vp=root/'vast/state/jobs'/(a['jobid']+'.json')
            if vp.exists():v=load(vp,'vast_job_state');job.update(worker_status=v['status'],tries=v.get('tries',0));job['active']=v['status']not in ('synced','failed','cancelled')
            else:job.update(worker_status='scheduler_state_not_observed',active=status not in ('complete','admitted','failed','retired_after_claim'))
        failure=record.get('error')or record.get('reason')or state.get('runtime_failure')or state.get('error');progress=max(float(v)for v in [pathlib.Path(path).stat().st_mtime,record.get('admitted_at',0),record.get('accepted_at',0),record.get('terminal_observed_at',0)]if isinstance(v,(int,float)))
        detail=[]
        if job and not job['active'] and status in ('failed','complete'):
            try:
                msg,evidence=failure_observation(job['job_dir'],root)
                if msg:detail=[{'reason':msg,'evidence':evidence}];progress=max([progress]+[x.get('mtime',0)for x in evidence])
            except (OSError,ValueError,KeyError,TypeError)as exc:errors.append({'source':job['job_dir'],'error':str(exc)})
        observations[key].append({'controller':worker['name'],'controller_state':worker['state'],'controller_config':worker['config'],
            'status':status,'seed_status':state.get('status'),'reason':failure,'configuration':ref(path),'runtime':job['runtime']if job else(seed or{}).get('snapshot')or cfg.get('snapshot')or cfg.get('admission_snapshot'),
            'bucket':(record.get('decision')or{}).get('bucket'),'job':job,'failures':detail,'last_progress_epoch':progress,'observed_at':iso(now)})
    for worker in workers:
        try:
            cfg=load(worker['config'],'controller_config');st=load(worker['state'],'controller_state')if pathlib.Path(worker['state']).exists()else{};split=cfg.get('admission_split')or cfg.get('source_split')
            if split in SPLITS:
                for cls in cfg.get('admitted_classes',[])if split=='eval_model'else cfg.get('admitted_tasks',[]):
                    if cls not in cfg.get('retired_classes',[]) and cls not in cfg.get('excluded_classes',[]):ownership[(split,cls)].append(worker['name'])
            seeds=list(cfg.get('seeds',[]))
            if cfg.get('seed_catalog'):seeds+=load(cfg['seed_catalog'],'seed_catalog').get('seeds',[])
            for seed in seeds:
                ss=st.get('seeds',{}).get(seed['id'],{});queues=ss.get('queues',[])+ss.get('production',[])
                observation({},worker,cfg,ss,seed)
                for r in queues[-3:]:observation(r,worker,cfg,ss,seed)
            for r in st.get('records',{}).values():
                if isinstance(r,dict)and r.get('admission_id'):observation(r,worker,cfg,st)
        except (OSError,ValueError,KeyError,TypeError)as exc:errors.append({'source':worker['name'],'error':type(exc).__name__+': '+str(exc)})
    engineering=root/'infra/heldout_recovery/engineering_actions.json'
    if engineering.exists():
        valid_scopes={(r['split'],r['class_id'])for r in rows}
        for entry in load(engineering)['rows']:
            key=(entry['split'],entry['class_id']);require(key in valid_scopes,'unknown engineering scope')
            source=entry['evidence'][0];require(file_digest(source['path'])==source['sha256'],'engineering evidence changed')
            records=entry.get('jobs')or[None]
            for job in records:
                if job:
                    job=dict(job,active=True,worker_status='scheduler_state_not_observed')
                    vp=root/'vast/state/jobs'/(job['job_id']+'.json')
                    if vp.exists():
                        v=load(vp);job.update(worker_status=v['status'],active=v['status']not in ('synced','failed','cancelled'))
                observations[key].append({'controller':'owned_native_engineering','controller_state':str(engineering),
                    'controller_config':str(engineering),'configuration':entry.get('configuration'),
                    'runtime':entry.get('runtime'),'bucket':entry.get('bucket'),'job':job,'failures':[],
                    'status':entry.get('status','native_preparation'),'reason':entry['blocker'],'next_action':entry['next_action'],
                    'last_progress_epoch':entry['observed_unix'],'observed_at':iso(now),'evidence':entry['evidence']})
    jobs=[o['job']for obs in observations.values()for o in obs if o['job']]
    sched,serrors=scheduler_states({j['job_id']for j in jobs if j['active']and j['worker_status']=='scheduler_state_not_observed'})
    errors.extend(serrors)
    for j in jobs:
        if j['job_id']in sched:
            j.update(sched[j['job_id']]);j['active']=j['worker_status']in ('RUNNING','PENDING','CONFIGURING','COMPLETING','SUSPENDED')
    for row in rows:
        key=(row['split'],row['class_id']);countkey=key+(row['bucket'],);obs=observations[key];same=[o for o in obs if o['bucket']is None or row['bucket']is None or o['bucket']==row['bucket']];jobs={o['job']['job_id']:o['job']for o in same if o['job']and o['job']['active']};failures=[f for o in same for f in o['failures']];latest=max(same,key=lambda o:o['last_progress_epoch'],default=None);n=counts[countkey];remain=max(0,row['target']-n)
        reason=(latest or{}).get('reason')or(latest or{}).get('seed_status')or(latest or{}).get('status');prep=prepared[key];state=recovery_state(remain,list(jobs.values()),latest)
        blocker=None if not remain else str(reason)if reason else 'No accepted exact-source admission/refill record; candidate preparation is not validation.'
        action='Preserve validated exact configuration; quota complete.'if not remain else'Follow active bounded job through its own strict replay and held-out acceptance.'if jobs else'Diagnose latest native/demo/replay rejection against exact candidate and dependencies before a fresh bounded recovery.'if failures else'Bind ready candidate to native geometry, demo and own strict replay, then admit only this held-out scope.'
        if state=='ready_for_scoped_refill':
            blocker='New exact runtime passed admission; remaining quota needs a scoped refill.'
            action='Author only the unfinished strict bucket from this admitted exact configuration, with its own new replay before acceptance.'
        elif state=='awaiting_acceptance':
            blocker='Collection is terminal; own fresh replay or strict acceptance reconciliation is pending.'
            action='Reconcile this exact allocation through its own physics gate and acceptance; do not duplicate sampling.'
        elif state=='awaiting_observation':
            blocker='Backend state unavailable; the submission receipt does not establish running or pending GPUs.'
        elif state=='unattempted'and prep:
            blocker=prep[-1]['stage']+': '+prep[-1]['reason']
            action='Inspect exact candidate and unmet native dependencies in '+', '.join(e['path']for e in prep[-1]['evidence'])+'; prepare a bounded native geometry proof before demo/replay admission.'
        if remain and (latest or{}).get('next_action'):
            action=latest['next_action']
        row.update(accepted=n,historical_accepted=rawcounts[countkey],remaining=remain,owner='/root/templates_assets',state=state,blocker=blocker,next_action=action,
            configuration=(latest or{}).get('configuration'),runtime=(latest or{}).get('runtime'),controller=sorted(set(ownership[key]+[o['controller']for o in obs])),active_jobs=list(jobs.values()),
            last_accepted_at=iso(last[countkey]),last_accepted_time_kind='immutable committed-group export mtime; exact commit timestamp not recorded'if last[countkey]else'unavailable',
            last_progress_at=iso(max([last[countkey]]+[o['last_progress_epoch']for o in same])),failures=failures,
            flat_progress_investigation={'status':'active_validation'if jobs else'validated_refill_needed'if state=='ready_for_scoped_refill'else'actionable_failure'if state=='failed'else'no_active_scope'if remain else'quota_complete',
                'controller_observations':same,'preparation':prep,'candidate_is_not_demo_or_replay_pass':True})
    summary={s:{'target':sum(r['target']for r in rows if r['split']==s),'accepted':sum(r['accepted']for r in rows if r['split']==s),
        'historical_accepted':sum(r['historical_accepted']for r in rows if r['split']==s),'remaining':sum(r['remaining']for r in rows if r['split']==s),
        'complete_buckets':sum(not r['remaining']for r in rows if r['split']==s),'total_buckets':sum(r['split']==s for r in rows)}for s in SPLITS}
    unmatched=[{'key':list(k),'accepted':v}for k,v in counts.items()if k not in {(r['split'],r['class_id'],r['bucket'])for r in rows}];require(not unmatched,'accepted held records fall outside frozen targets: '+str(unmatched))
    value={'schema':'isg-heldout-recovery-ledger-v1','campaign_id':'campaign_20261110','observed_at':iso(now),'owner':'/root/templates_assets','summary':summary,'rows':rows,
        'sources':sources+snapshot_sources,'current_eligibility_ledger':lref,'errors':errors,'training_generation_authorized':False,
        'semantics':'All99 model classes and23 category/room task scopes enumerated from frozen100-class inventory and exact owner quotas. Bucket targets remain strict. Candidate preparation is separate from native/demo/replay/accepted proof. No collection action is performed by this audit.'}
    write_json_atomic(D/'ledger.json',value);print(json.dumps({'path':str(D/'ledger.json'),'summary':summary,'errors':len(errors),'rows':len(rows)}));return value

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=pathlib.Path,required=True);a=p.parse_args();publish(a.root)
if __name__=='__main__':main()
