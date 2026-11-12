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
from .contracts import digest,file_digest,require,write_json_atomic
from .training_stream import native_evidence
from .training_canaries import publish_canary
from .training_restoration import checked,validated_canary
from .training_batch_probe import inputs,verify_runtime_compatibility
from .training_sequential_augmentation import PHYSICS,RENDER,verify,verify_render_cleanup


def read(path):return json.loads(Path(path).read_text())
def ref(path):return {'path':str(Path(path).resolve()),'sha256':file_digest(path)}
def immutable(path,value):
    path=Path(path)
    if path.exists():require(read(path)==json.loads(json.dumps(value)),'immutable sequential artifact differs: '+str(path))
    else:path.parent.mkdir(parents=True,exist_ok=True);write_json_atomic(path,value)
    return ref(path)


def prepare_matrix(root,batch,reviews,runtime,started_at,render_runtime=None,physics_probe_runtime=None):
    root,batch,runtime=Path(root),Path(batch),Path(runtime)
    render_runtime=Path(render_runtime)if render_runtime else runtime
    physics_probe_runtime=Path(physics_probe_runtime)if physics_probe_runtime else runtime
    directory=batch/'sequential_verification_r1';manifest=directory/'requests.json'
    if manifest.exists():
        saved=read(manifest)
        require(saved['canaries']==reviews and saved['runtime']==str(runtime)and saved.get('render_runtime',str(runtime))==str(render_runtime)and
                saved.get('physics_probe_runtime',str(runtime))==str(physics_probe_runtime),'matrix source identity changed')
        for row in saved['records']:
            checked(root,row['request']);checked(root,row['preflight'])
        return ref(manifest)
    sources=[validated_canary(root,r,started_at) for r in reviews]
    require(len(sources)==2 and sources[0]['instance_id']!=sources[1]['instance_id'],'matrix needs two own-passed inputs')
    stamp=runtime/'.snapshot_commit';native_hash=file_digest(stamp)
    require(all(s['export_record']['validation_key']['code_sha256']==native_hash for s in sources),'matrix native source differs')
    if physics_probe_runtime!=runtime:
        proof=render_runtime_proof(directory/'runtime_compatibility.json',runtime,physics_probe_runtime,physics_checker=True)
    else:
        payloads={str(p.relative_to(runtime)):file_digest(p) for folder in ('ispatialgym','scripts') for p in sorted((runtime/folder).rglob('*.py')) if p.is_file()}
        proof=immutable(directory/'runtime_compatibility.json',{'schema':'isg-exact-native-runtime-compatibility-v1','native_runtime':ref(stamp),'python_payloads':payloads,'physical_code_changes':0})
    render_proof=render_runtime_proof(directory/'renderer_runtime_compatibility.json',runtime,render_runtime)if render_runtime!=runtime else proof
    env=ref(Path(sources[0]['source_job'])/'job.env');rows=[]
    for stage,specs in [('physics',PHYSICS),('rendering',RENDER)]:
        kind='physics_action_open_loop'if stage=='physics'else'kinematic_observation_rendering'
        stage_runtime=physics_probe_runtime if stage=='physics'else render_runtime
        stage_proof=proof if stage=='physics'else render_proof
        verified_stage_inputs=None
        for name,(control,order)in specs.items():
            destination=directory/stage/name;request=destination/'request.json';output=destination/'output'
            value={'schema':'isg-training-batch-probe-request-v1','campaign_id':'campaign_20261110','canaries':reviews,
                'sequential_augmentation':True,'probe_runtime_sha256':file_digest(stage_runtime/'.snapshot_commit'),'runtime_compatibility':stage_proof,
                'observe_reset_boundary':True,'observe_final_pre_action':True,'observe_native_dynamics':True,
                'observe_next_native_control_step':True,'accepted':False,'eligible_for_benchmark':False,
                'benchmark_credit':0,'quota_consumed':0,'production_released':False,'source_files_unchanged':True,
                'scope':'Changed lighting and one fixed non-task actor parking; exact independent references and native cleanup controls; one actual world constructor, no added physics steps.'}
            if control:value.update(diagnostic_control=control,diagnostic_only=True)
            if stage=='rendering'and render_runtime!=runtime:value['sequential_render_only']=True
            if stage=='physics'and physics_probe_runtime!=runtime:value['sequential_checker_only']=True
            request_ref=immutable(request,value)
            actual,records=inputs(root,request)
            # The frozen runtime and source bytes are common to this stage. Keep
            # every request/source binding check, but audit the full payload
            # inventory once here. Native entry and final verification still
            # independently perform their complete runtime checks per control.
            stage_inputs=digest({'request':{k:v for k,v in actual.items()
                if k not in ('diagnostic_control','diagnostic_only')},'records':records})
            require(actual.get('diagnostic_control')==control and
                    actual.get('diagnostic_only',False)==bool(control),
                    'matrix control scope changed')
            if verified_stage_inputs is None:
                verify_runtime_compatibility(root,actual,records,stage_runtime)
                verified_stage_inputs=stage_inputs
            else:require(stage_inputs==verified_stage_inputs,'matrix stage input identity changed')
            preflight=immutable(destination/'preflight.json',{'passed':True,'request':request_ref,'runtime':str(stage_runtime),
                'runtime_proof':stage_proof,'source_environment':env,'control':name,'kind':kind,'max_native_episodes':len(order),
                'output':str(output),'native_draws':0,'benchmark_credit':0,'actual_native_execution':False,
                'cpu_scope':'Exact source artifacts, recorded input contract and full frozen payloads; native saved-world audit still executes before world creation.'})
            rows.append({'control':name,'stage':stage,'kind':kind,'request':request_ref,'runtime':str(stage_runtime),
                'runtime_proof':stage_proof,'source_environment':env,'output':str(output),'preflight':preflight,
                'max_native_episodes':len(order),'new_native_draws':0,'benchmark_credit':0})
    return immutable(manifest,{'schema':'isg-sequential-augmentation-dispatch-v1','campaign_id':'campaign_20261110',
        'runtime':str(runtime),'physics_probe_runtime':str(physics_probe_runtime),'render_runtime':str(render_runtime),'canaries':reviews,'records':rows,'new_native_draws':0,'benchmark_credit':0,'production_released':False})


def render_runtime_proof(path,native,runtime,*,physics_checker=False):
    if Path(path).exists():
        # This is an immutable candidate proof, not authorization by itself.
        # Its full payload inventory is re-audited by the caller before use.
        saved=read(path)
        require(saved.get('schema')==('isg-sequential-checker-runtime-compatibility-v1'if physics_checker else'isg-render-cleanup-runtime-compatibility-v1')and
                saved.get('native_runtime')==ref(native/'.snapshot_commit')and
                saved.get('checker_runtime'if physics_checker else'cleanup_runtime')==ref(runtime/'.snapshot_commit'),
                'saved runtime proof identity changed')
        return ref(path)
    changed={}
    for folder in ('ispatialgym','scripts'):
        paths={str(p.relative_to(tree))for tree in (native,runtime)for p in(tree/folder).rglob('*.py')}
        for relative in paths:
            a,b=native/relative,runtime/relative
            if not a.is_file()or not b.is_file()or file_digest(a)!=file_digest(b):changed[relative]=file_digest(b)
    value={'schema':'isg-sequential-checker-runtime-compatibility-v1'if physics_checker else'isg-render-cleanup-runtime-compatibility-v1',
        'native_runtime':ref(native/'.snapshot_commit'),('checker_runtime'if physics_checker else'cleanup_runtime'):ref(runtime/'.snapshot_commit'),
        'physical_code_changes':0,'changed_payloads':changed,'permitted_execution_kinds':['physics_action_open_loop'if physics_checker else'kinematic_observation_rendering']}
    relative='ispatialgym/oracle/campaign_batch.py'
    if relative in changed:value['saved_scene_checker']={'base_sha256':file_digest(native/relative),'runtime_sha256':file_digest(runtime/relative)}
    return immutable(path,value)


def prepare_render_cleanup(root,batch,reviews,native,runtime,source_environment):
    root,batch,native,runtime=map(Path,(root,batch,native,runtime))
    directory=batch/'renderer_cleanup_r1';manifest=directory/'requests.json'
    if manifest.exists():
        saved=read(manifest);require(saved['canaries']==reviews and saved['runtime']==str(runtime),'renderer cleanup source changed');return ref(manifest)
    proof=render_runtime_proof(directory/'runtime_compatibility.json',native,runtime)
    rows=[]
    for name,control,boundary in [('failure','RENDER_FAILURE_A','writer'),('interruption','RENDER_INTERRUPT_A','post_render')]:
        D=directory/name;request=immutable(D/'request.json',{'schema':'isg-training-batch-probe-request-v1',
            'campaign_id':'campaign_20261110','canaries':reviews,'probe_runtime_sha256':file_digest(runtime/'.snapshot_commit'),
            'runtime_compatibility':proof,'sequential_augmentation':True,'render_cleanup_only':True,
            'diagnostic_control':control,'render_fault_boundary':boundary,'diagnostic_only':True,
            'observe_native_dynamics':True,'accepted':False,'eligible_for_benchmark':False,'production_released':False,
            'benchmark_credit':0,'quota_consumed':0,'source_files_unchanged':True})
        value,records=inputs(root,request['path']);verify_runtime_compatibility(root,value,records,runtime)
        preflight=immutable(D/'preflight.json',{'passed':True,'request':request,'runtime':str(runtime),
            'runtime_proof':proof,'source_environment':source_environment,'kind':'kinematic_observation_rendering',
            'control':name,'max_native_episodes':2,'output':str(D/'output'),'native_draws':0,'benchmark_credit':0,
            'actual_native_execution':False,'cpu_scope':'Exact source bytes and minimal renderer-only compatibility; native saved-scene audit still required.'})
        rows.append({'control':name,'stage':'rendering','kind':'kinematic_observation_rendering','request':request,
            'runtime':str(runtime),'runtime_proof':proof,'source_environment':source_environment,'output':str(D/'output'),
            'preflight':preflight,'max_native_episodes':2,'new_native_draws':0,'benchmark_credit':0})
    return immutable(manifest,{'schema':'isg-sequential-render-cleanup-dispatch-v1','campaign_id':'campaign_20261110',
        'runtime':str(runtime),'canaries':reviews,'records':rows,'new_native_draws':0,'benchmark_credit':0,'production_released':False})


def publish_snapshot(path,value):
    path=Path(path)
    if path.exists():
        old=path.read_bytes();history=path.parent/(path.stem+'_history');history.mkdir(exist_ok=True)
        old_path=history/(file_digest(path)+'.json')
        if not old_path.exists():old_path.write_bytes(old)
    write_json_atomic(path,value)


def publish_reviews(path,value):
    """Keep superseded source verdicts visible with their original observation time."""
    previous=read(path) if Path(path).exists() else {}
    records={key:{**record,'observed_at':record.get('observed_at',previous.get('observed_at')),
                  'current_pair_member':False}
             for key,record in previous.get('records',{}).items() if key not in value['records']}
    records.update({key:{**record,'observed_at':value['observed_at'],'current_pair_member':True}
                    for key,record in value['records'].items()})
    publish_snapshot(path,{**value,'records':records})


def replay_failure(root,gate_path,gate):
    """Retain a failed gate's measured reason without converting task success to QA success."""
    require(gate.get('success') is False,'expected explicit failed own replay')
    quality=gate.get('native_quality',{});result=gate.get('result',{})
    evidence={'gate':ref(gate_path),'validated_at':gate.get('validated_at'),
        'reasons':quality.get('reasons',[]),'hard_qa':result.get('qa_hard',[]),
        'geometric_success':result.get('geometric_success'),
        'symbolic_success':result.get('symbolic_success'),
        'action_steps':result.get('n_action_steps'),'replay_returncode':gate.get('replay_rc')}
    if quality.get('native_report'):
        reference={'path':quality['native_report'],'sha256':quality['native_report_sha256']}
        report=checked(root,reference)
        evidence.update(native_report=reference,goal_occlusion=report.get('qa',{}).get('metrics',{}).get('goal_occlusion'))
    return evidence


def selected_requests(root,configuration):
    """A prospective pair can retain passed A and select one explicit B retry."""
    if not configuration.get('members'):
        document=checked(root,configuration['requests'])
        return document,[configuration['requests']for _ in document['records']]
    require(len(configuration['members'])==2,'exact pair selection required')
    records=[];references=[]
    for member in configuration['members']:
        document=checked(root,member['requests'])
        require(document['campaign_id']=='campaign_20261110','foreign pair source')
        matches=[r for r in document['records']if r['candidate_id']==member['candidate_id']]
        require(len(matches)==1,'ambiguous prospective pair member')
        records.append(matches[0]);references.append(member['requests'])
    require(records[0]['candidate_id']!=records[1]['candidate_id']and records[0]['start']!=records[1]['start'],
            'prospective pair reused an input identity')
    return {'campaign_id':'campaign_20261110','records':records},references


def run_cycle(root,configuration,state):
    root=Path(root);batch=Path(configuration['batch']);requests,source_requests=selected_requests(root,configuration)
    require(requests['campaign_id']=='campaign_20261110'and len(requests['records'])==2,'not exact two-input scope')
    prepared=Path(configuration['prepared'])
    authdir=Path(configuration['authorizations']);runtime=Path(configuration['runtime']);started=configuration['campaign_started_at']
    out={'schema':'isg-existing-training-canary-observation-v1','campaign_id':'campaign_20261110',
         'observed_at':datetime.now(timezone.utc).isoformat(),'source_request':configuration['requests'],
         'records':{},'new_native_draws':0,'benchmark_credit':0}
    launchpath=prepared/'launched.json';launched=read(launchpath).get('jobs',[])if launchpath.exists()else[]
    for row,source_request in zip(requests['records'],source_requests):
        cid=row['candidate_id'];value={'candidate_id':cid,'status':'awaiting_native','native_written':0,'benchmark_credit':0,'new_native_draws':0}
        try:
            authorizer_ref=configuration.get('authorizers',{}).get(cid,configuration['authorizer'])
            authorizer=Path(authorizer_ref['path']);require(file_digest(authorizer)==authorizer_ref['sha256'],'collector authorizer changed')
            evidence=native_evidence(row,root)
            if evidence is None:out['records'][cid]=value;continue
            value.update(native_written=1,native_evidence=evidence,status='awaiting_preserved_input_collector_preparation')
            prepared_ref=prepared/(cid+'.json')
            if not prepared_ref.exists():out['records'][cid]=value;continue
            copied=read(prepared_ref)
            require(copied['candidate_id']==cid and copied['start']==row['start']and copied['count']==1 and
                    copied['configuration']=={'path':row['config_path'],'sha256':row['config_sha256']}and
                    copied['runtime']==row['native_runtime']and copied['source_request']==source_request,
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
                g=read(gate)
                if g.get('success') is False:
                    value.update(status='failed_own_replay',failure_evidence=replay_failure(root,gate,g),
                        next_action='Diagnose this exact failed QA or replay evidence before a changed bounded input; no automatic native retry')
                    out['records'][cid]=value
                    continue
                require(g.get('success')is True,'own physics replay has no explicit verdict')
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
        publish_reviews(batch/'existing_native_reviews.json',out)
    publish_reviews(batch/'existing_native_reviews.json',out)
    for source_request in source_requests:
        parent=Path(source_request['path']).parent
        if parent==batch:continue
        candidates={r['candidate_id']for r,source in zip(requests['records'],source_requests)if source==source_request}
        publish_reviews(parent/'existing_native_reviews.json',{**out,'source_request':source_request,
            'records':{k:v for k,v in out['records'].items()if k in candidates}})
    state.update(records=out['records'],observed_at=out['observed_at'],pid=os.getpid(),owner='/root/dataset_contract',new_native_draws=0,benchmark_credit=0)
    if all(v['status']=='validated_canary_exported'for v in out['records'].values())and len(out['records'])==2:
        reviews=[{'path':out['records'][r['candidate_id']]['validation_review']['review'],'sha256':out['records'][r['candidate_id']]['validation_review']['sha256']}for r in requests['records']]
        try:
            state['matrix_request']=prepare_matrix(root,batch,reviews,runtime,started,configuration.get('render_cleanup_runtime'),configuration.get('physics_probe_runtime'));state.pop('matrix_error',None)
            if configuration.get('render_cleanup_runtime'):
                env=checked(root,state['matrix_request'])['records'][0]['source_environment']
                state['renderer_cleanup_request']=prepare_render_cleanup(root,batch,reviews,runtime,configuration['render_cleanup_runtime'],env)
        except Exception as error:state['matrix_error']=f'{type(error).__name__}: {error}'
    if state.get('matrix_request'):
        manifest=checked(root,state['matrix_request']);matrix={'schema':'isg-sequential-augmentation-verification-input-v1','physics':{},'rendering':{},'execution':{'physics':{},'rendering':{}}};progress=[]
        records=manifest['records'];verification_directory=batch/'sequential_verification_r1'
        if configuration.get('physics_followup'):
            followup=checked(root,configuration['physics_followup'])
            require(followup['canaries']==manifest['canaries']and len(followup['records'])==len(PHYSICS)and
                    {r['control']for r in followup['records']}==set(PHYSICS)and
                    all(r['stage']=='physics'and checked(root,r['request']).get('sequential_anchor_rebase')is True
                        for r in followup['records']),'prospective physics followup identity differs')
            records=[r for r in records if r['stage']=='rendering']+followup['records']
            verification_directory=Path(configuration['physics_followup']['path']).parent
            matrix['prior_cold_references']={name:{'probe':ref(batch/'sequential_verification_r1/physics'/name/'output/probe.json'),
                'execution':ref(batch/'sequential_verification_r1/physics'/name/'execution_observation.json')}
                for name in('fresh_A','fresh_B')}
            state['physics_followup']=configuration['physics_followup']
        for row in records:
            directory=Path(row['request']['path']).parent;probe=Path(row['output'])/'probe.json';execution=directory/'execution_observation.json'
            progress.append({'control':row['control'],'stage':row['stage'],'probe':ref(probe)if probe.exists()else None,'execution':ref(execution)if execution.exists()else None})
            if probe.exists()and execution.exists():matrix[row['stage']][row['control']]=ref(probe);matrix['execution'][row['stage']][row['control']]=ref(execution)
        state['matrix_progress']=progress
        if len(matrix['physics'])==len(PHYSICS)and len(matrix['rendering'])==len(RENDER)and not state.get('verification'):
            try:
                inp=immutable(verification_directory/'verification_inputs.json',matrix)
                state['verification']=immutable(verification_directory/'verification.json',verify(root,inp,started));state.pop('verification_error',None)
            except Exception as error:state['verification_error']=f'{type(error).__name__}: {error}'
    if state.get('renderer_cleanup_request') and not state.get('renderer_cleanup_verification'):
        try:
            m=checked(root,state['renderer_cleanup_request']);controls={}
            for row in m['records']:
                D=Path(row['request']['path']).parent;probe=Path(row['output'])/'probe.json';execution=D/'execution_observation.json'
                if probe.exists()and execution.exists():controls[row['control']]={'probe':ref(probe),'execution':ref(execution)}
            fresh=batch/'sequential_verification_r1/rendering/fresh_A'
            if len(controls)==2 and(fresh/'output/probe.json').exists()and(fresh/'execution_observation.json').exists():
                document={'schema':'isg-sequential-render-cleanup-input-v1','controls':controls,
                    'independent_A':{'probe':ref(fresh/'output/probe.json'),'execution':ref(fresh/'execution_observation.json')}}
                inp=immutable(batch/'renderer_cleanup_r1/verification_inputs.json',document)
                state['renderer_cleanup_verification']=immutable(batch/'renderer_cleanup_r1/verification.json',verify_render_cleanup(root,inp,started))
        except Exception as error:state['renderer_cleanup_error']=f'{type(error).__name__}: {error}'
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
