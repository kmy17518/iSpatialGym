"""Advance two exact changed-augmentation inputs without new sampling.

The compute owner publishes copied collector shells and launches registered
shells. This owner registers each exact copy, publishes its own strict canary,
and authors the bounded independent-reference matrix. It never submits GPUs.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import fcntl,json,os,subprocess,time
from pathlib import Path
from .contracts import file_digest,require,write_json_atomic
from .training_stream import native_evidence
from .training_canaries import publish_canary
from .training_restoration import checked,validated_canary
from .training_batch_probe import inputs,verify_runtime_compatibility
from .training_sequential_augmentation import PHYSICS,RENDER,verify


def read(path):return json.loads(Path(path).read_text())
def ref(path):return {'path':str(Path(path).resolve()),'sha256':file_digest(path)}
def immutable(path,value):
    path=Path(path)
    if path.exists():require(read(path)==json.loads(json.dumps(value)),'immutable sequential artifact differs: '+str(path))
    else:path.parent.mkdir(parents=True,exist_ok=True);write_json_atomic(path,value)
    return ref(path)


def prepare_matrix(root,batch,reviews,runtime,started_at):
    root,batch,runtime=Path(root),Path(batch),Path(runtime)
    directory=batch/'sequential_verification_r1';manifest=directory/'requests.json'
    if manifest.exists():
        saved=read(manifest)
        require(saved['canaries']==reviews and saved['runtime']==str(runtime),'matrix source identity changed')
        for row in saved['records']:
            checked(root,row['request']);checked(root,row['preflight'])
        return ref(manifest)
    sources=[validated_canary(root,r,started_at) for r in reviews]
    require(len(sources)==2 and sources[0]['instance_id']!=sources[1]['instance_id'],'matrix needs two own-passed inputs')
    stamp=runtime/'.snapshot_commit';native_hash=file_digest(stamp)
    require(all(s['export_record']['validation_key']['code_sha256']==native_hash for s in sources),'matrix native source differs')
    payloads={str(p.relative_to(runtime)):file_digest(p) for folder in ('ispatialgym','scripts') for p in sorted((runtime/folder).rglob('*.py')) if p.is_file()}
    proof=immutable(directory/'runtime_compatibility.json',{'schema':'isg-exact-native-runtime-compatibility-v1','native_runtime':ref(stamp),'python_payloads':payloads,'physical_code_changes':0})
    env=ref(Path(sources[1]['source_job'])/'job.env');rows=[]
    for stage,specs in [('physics',PHYSICS),('rendering',RENDER)]:
        kind='physics_action_open_loop'if stage=='physics'else'kinematic_observation_rendering'
        for name,(control,order)in specs.items():
            destination=directory/stage/name;request=destination/'request.json';output=destination/'output'
            value={'schema':'isg-training-batch-probe-request-v1','campaign_id':'campaign_20261110','canaries':reviews,
                'sequential_augmentation':True,'probe_runtime_sha256':native_hash,'runtime_compatibility':proof,
                'observe_reset_boundary':True,'observe_final_pre_action':True,'observe_native_dynamics':True,
                'observe_next_native_control_step':True,'accepted':False,'eligible_for_benchmark':False,
                'benchmark_credit':0,'quota_consumed':0,'production_released':False,'source_files_unchanged':True,
                'scope':'Changed lighting and one fixed non-task actor parking; exact independent references and native cleanup controls; one actual world constructor, no added physics steps.'}
            if control:value.update(diagnostic_control=control,diagnostic_only=True)
            request_ref=immutable(request,value)
            actual,records=inputs(root,request);verify_runtime_compatibility(root,actual,records,runtime)
            preflight=immutable(destination/'preflight.json',{'passed':True,'request':request_ref,'runtime':str(runtime),
                'runtime_proof':proof,'source_environment':env,'control':name,'kind':kind,'max_native_episodes':len(order),
                'output':str(output),'native_draws':0,'benchmark_credit':0,'actual_native_execution':False,
                'cpu_scope':'Exact source artifacts, recorded input contract and full frozen payloads; native saved-world audit still executes before world creation.'})
            rows.append({'control':name,'stage':stage,'kind':kind,'request':request_ref,'runtime':str(runtime),
                'runtime_proof':proof,'source_environment':env,'output':str(output),'preflight':preflight,
                'max_native_episodes':len(order),'new_native_draws':0,'benchmark_credit':0})
    return immutable(manifest,{'schema':'isg-sequential-augmentation-dispatch-v1','campaign_id':'campaign_20261110',
        'runtime':str(runtime),'canaries':reviews,'records':rows,'new_native_draws':0,'benchmark_credit':0,'production_released':False})


def publish_snapshot(path,value):
    path=Path(path)
    if path.exists():
        old=path.read_bytes();history=path.parent/(path.stem+'_history');history.mkdir(exist_ok=True)
        old_path=history/(file_digest(path)+'.json')
        if not old_path.exists():old_path.write_bytes(old)
    write_json_atomic(path,value)


def run_cycle(root,configuration,state):
    root=Path(root);batch=Path(configuration['batch']);requests=checked(root,configuration['requests'])
    require(requests['campaign_id']=='campaign_20261110'and len(requests['records'])==2,'not exact two-input scope')
    prepared=Path(configuration['prepared']);authorizer=Path(configuration['authorizer']['path'])
    require(file_digest(authorizer)==configuration['authorizer']['sha256'],'collector authorizer changed')
    authdir=Path(configuration['authorizations']);runtime=Path(configuration['runtime']);started=configuration['campaign_started_at']
    out={'schema':'isg-existing-training-canary-observation-v1','campaign_id':'campaign_20261110',
         'observed_at':datetime.now(timezone.utc).isoformat(),'source_request':configuration['requests'],
         'records':{},'new_native_draws':0,'benchmark_credit':0}
    launchpath=prepared/'launched.json';launched=read(launchpath).get('jobs',[])if launchpath.exists()else[]
    for row in requests['records']:
        cid=row['candidate_id'];value={'candidate_id':cid,'status':'awaiting_native','native_written':0,'benchmark_credit':0,'new_native_draws':0}
        try:
            evidence=native_evidence(row,root)
            if evidence is None:out['records'][cid]=value;continue
            value.update(native_written=1,native_evidence=evidence,status='awaiting_preserved_input_collector_preparation')
            prepared_ref=prepared/(cid+'.json')
            if not prepared_ref.exists():out['records'][cid]=value;continue
            copied=read(prepared_ref)
            require(copied['candidate_id']==cid and copied['start']==row['start']and copied['count']==1 and
                    copied['configuration']=={'path':row['config_path'],'sha256':row['config_sha256']}and
                    copied['runtime']==row['native_runtime']and copied['source_request']==configuration['requests'],
                    'collector prepared identity differs')
            auth=authdir/cid/'authorization_receipt.json'
            if not auth.exists():
                log=batch/'stream'/('registration_'+cid+'_'+str(time.time_ns())+'.log');log.parent.mkdir(exist_ok=True)
                with log.open('x')as stream:
                    result=subprocess.run([os.sys.executable,str(authorizer),str(prepared_ref),file_digest(prepared_ref)],stdout=stream,stderr=subprocess.STDOUT,timeout=900)
                require(result.returncode==0,'collector registration failed: '+str(log))
            receipt=read(auth)
            require(receipt['prepared']==ref(prepared_ref)and receipt['job_dir']==copied['job_dir'],'collector registration differs')
            value.update(status='awaiting_collection_allocation',authorization_receipt=ref(auth),job_dir=copied['job_dir'])
            matches=[x for x in launched if x.get('candidate',x.get('candidate_id'))==cid]
            require(len(matches)<=1,'multiple collector allocations need explicit recovery')
            if not matches:out['records'][cid]=value;continue
            allocation=matches[0];job=Path(allocation['job_dir']);require(str(job)==copied['job_dir']and allocation['original_index']==row['start'],'collector allocation differs')
            value['allocation']={'start':str(row['start']),'n':'1','jobid':str(allocation['job_id']),'job_dir':str(job),'key':row['class_id'],'code':str(runtime),'launch':ref(launchpath)}
            value['status']='awaiting_demo_observation'
            stats=job/'job_stats.json'
            if stats.exists():value.update(collector_stats=ref(stats),collector_summary=read(stats))
            reports=[read(p)for p in(job/'demo').glob('*/meta.json')]
            value.update(demo_completed=len(reports),demo_successful=sum(r.get('success')is True and not r.get('qa',{}).get('hard_triggered')for r in reports))
            if reports:value['status']='awaiting_own_replay'if value['demo_successful']else'failed_demo'
            gate=root/'validation/open_loop'/row['class_id']/str(allocation['job_id'])/'gate.json';value['own_replay_gate']=str(gate)
            if gate.exists():
                g=read(gate);require(g.get('success')is True,'own physics replay failed')
                previous=state.get('records',{}).get(cid,{})
                if previous.get('gate_sha256')==file_digest(gate) and previous.get('validation_review'):
                    review=previous['validation_review'];checked(root,{'path':review['review'],'sha256':review['sha256']})
                else:review=publish_canary(job,gate,root=root,started_at=started)
                value.update(status='validated_canary_exported',validation_review=review,gate_sha256=file_digest(gate))
        except Exception as error:
            waiting=isinstance(error,FileExistsError)and '.lockdir'in str(error)
            value.update(status='waiting_publication_lock'if waiting else'engineering_blocked',error=f'{type(error).__name__}: {error}',
                next_action='Retry exact CPU stage after current lock owner exits'if waiting else'Inspect exact recorded cause before any retry; no new native draw')
        out['records'][cid]=value
        publish_snapshot(batch/'existing_native_reviews.json',out)
    publish_snapshot(batch/'existing_native_reviews.json',out)
    state.update(records=out['records'],observed_at=out['observed_at'],pid=os.getpid(),owner='/root/dataset_contract',new_native_draws=0,benchmark_credit=0)
    if all(v['status']=='validated_canary_exported'for v in out['records'].values())and len(out['records'])==2:
        reviews=[{'path':out['records'][r['candidate_id']]['validation_review']['review'],'sha256':out['records'][r['candidate_id']]['validation_review']['sha256']}for r in requests['records']]
        try:state['matrix_request']=prepare_matrix(root,batch,reviews,runtime,started);state.pop('matrix_error',None)
        except Exception as error:state['matrix_error']=f'{type(error).__name__}: {error}'
    if state.get('matrix_request'):
        manifest=checked(root,state['matrix_request']);matrix={'schema':'isg-sequential-augmentation-verification-input-v1','physics':{},'rendering':{},'execution':{'physics':{},'rendering':{}}};progress=[]
        for row in manifest['records']:
            directory=Path(row['request']['path']).parent;probe=Path(row['output'])/'probe.json';execution=directory/'execution_observation.json'
            progress.append({'control':row['control'],'stage':row['stage'],'probe':ref(probe)if probe.exists()else None,'execution':ref(execution)if execution.exists()else None})
            if probe.exists()and execution.exists():matrix[row['stage']][row['control']]=ref(probe);matrix['execution'][row['stage']][row['control']]=ref(execution)
        state['matrix_progress']=progress
        if len(matrix['physics'])==len(PHYSICS)and len(matrix['rendering'])==len(RENDER)and not state.get('verification'):
            try:
                inp=immutable(batch/'sequential_verification_r1/verification_inputs.json',matrix)
                state['verification']=immutable(batch/'sequential_verification_r1/verification.json',verify(root,inp,started))
            except Exception as error:state['verification_error']=f'{type(error).__name__}: {error}'
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--configuration',type=Path,required=True);p.add_argument('--interval',type=float,default=30);p.add_argument('--once',action='store_true');a=p.parse_args()
    configuration=read(a.configuration);directory=Path(configuration['batch'])/'stream';directory.mkdir(exist_ok=True)
    with (directory/'owner.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=read(directory/'state.json')if(directory/'state.json').exists()else{}
        while True:
            try:run_cycle(a.root,configuration,state);state.pop('cycle_error',None)
            except Exception as error:state.update(cycle_error=f'{type(error).__name__}: {error}',observed_at=datetime.now(timezone.utc).isoformat())
            publish_snapshot(directory/'state.json',state)
            print(json.dumps({'observed_at':state.get('observed_at'),'records':{k:v['status']for k,v in state.get('records',{}).items()},'matrix_request':state.get('matrix_request'),'cycle_error':state.get('cycle_error')}),flush=True)
            if a.once:break
            time.sleep(a.interval)


if __name__=='__main__':main()
