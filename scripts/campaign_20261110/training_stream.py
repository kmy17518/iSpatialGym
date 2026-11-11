"""Advance each completed native training canary to a bounded annotated request.

This CPU observer never launches GPU jobs. It emits immutable launch requests
and registers their exact one-instance authorization, independently per source.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import sys
import time
from scripts.campaign_20261110.contracts import ObjectRef,Template,ValidationKey,file_digest,require,write_json_atomic
from scripts.campaign_20261110.controller_families import reserve_indices
from scripts.campaign_20261110.materialize_training_table import bind_training_candidate
from scripts.campaign_20261110.production_gate import check_environment
from scripts.campaign_20261110.reservation_policy import active_policy
from scripts.campaign_20261110.storage import atomic_directory_lock


def read(path):return json.loads(Path(path).read_text())


def native_evidence(row,root):
    root=Path(root).resolve();output=Path(row['output']).resolve()
    require(output.is_relative_to(root),'native canary outside campaign')
    path=output/'sampling_gate.json'
    if not path.exists():return None
    gate=read(path)
    require(gate['campaign_id']=='campaign_20261110' and gate['candidate_id']==row['candidate_id']
            and gate['config_sha256']==row['config_sha256'],'native canary gate identity differs')
    require(gate['returncode']==0 and gate['tro_count']==1 and len(gate['tros'])==1,'native canary did not write exactly one successful sample')
    require(gate['gate']['allowed'] is True and gate['gate']['category']=='validation_canary','native stage lacks explicit authorization')
    require(gate['native_runtime']==row['native_runtime'],'native canary runtime differs')
    for item in gate['tros']:
        tro=Path(item['path']).resolve()
        require(tro.is_relative_to(output) and file_digest(tro)==item['sha256'],'native canary TRO changed')
        raw=read(tro)['ispatialgym'];index=int(raw['instance_id'].rsplit('__',1)[-1])
        require(index==row['start'],'native canary index differs')
    metadata=tro.parent.parent/'template_meta.json'
    require(metadata.exists() and read(metadata)['n_written']==1,'native canary template missing')
    return {'gate':str(path),'gate_sha256':file_digest(path),'metadata':str(metadata),'metadata_sha256':file_digest(metadata)}


def register(root,entry):
    root=Path(root);path=root/'controllers/training_production_gate.json'
    with atomic_directory_lock(path.with_suffix('.lockdir')):
        policy=read(path)
        require(policy['mode']=='blocked' and policy['standalone_training_allowed'] is False,'canary registration cannot change production release')
        same=[a for a in policy.get('allowed_canaries',[]) if (a['config_sha256'],a['start'],a['count'])==(entry['config_sha256'],entry['start'],entry['count'])]
        if same:require(same==[entry],'canary authorization conflicts');return
        history=root/'infra/training_production_gate/canary_stream_history';history.mkdir(parents=True,exist_ok=True)
        previous=history/(str(time.time_ns())+'.json');previous.write_bytes(path.read_bytes())
        policy.setdefault('allowed_canaries',[]).append(entry);policy['observed_at']=datetime.now(timezone.utc).isoformat()
        from .training_policy_history import replace_policy
        replace_policy(root, policy, expected_previous_sha256=file_digest(previous))


def prepare_annotated(row, *, root):
    root=Path(root).resolve();proof=native_evidence(row,root)
    if proof is None:return None
    for p,s in [('config_path','config_sha256'),('input_manifest','input_manifest_sha256'),('authorization_path','authorization_sha256')]:
        require(Path(row[p]).resolve().is_relative_to(root) and file_digest(row[p])==row[s],'native stage input changed: '+p)
    output=Path(row['config_path']).parent/('annotated_'+row['candidate_id'])
    if (output/'request.json').exists():
        previous=read(output/'request.json');require(previous['native_proof']==proof,'prepared native proof changed');return previous
    require(not output.exists(),'partial annotated preparation needs explicit recovery')
    cfg=read(row['config_path']);job=read(row['input_manifest'])['jobs'][row['manifest_index']]
    require(job['candidate_id']==row['candidate_id'] and job['production_released'] is False,'native job identity differs')
    original=Template(**{**cfg['template'],'objects':tuple(ObjectRef(**o)for o in cfg['template']['objects'])})
    active=active_policy(root);require(active is not None,'no active reservations')
    native=Path(row['native_runtime']);bound,perturbations=bind_training_candidate(original,job,Path(proof['metadata']),native,active[1],active[2],active[0])
    key=ValidationKey(bound.fingerprint,file_digest(native/'.snapshot_commit'),cfg['validation_key']['physics_sha256'],cfg['validation_key']['data_sha256'],perturbations.scale_fingerprint)
    token='training_height_annotated_'+row['candidate_id']+'_'+proof['gate_sha256'][:12];index=reserve_indices(root,token,1)
    output.mkdir();config_path=output/'campaign_config.json'
    config={'campaign_id':'campaign_20261110','split':'train','purpose':'training_pipeline_canary',
            'template':asdict(bound),'perturbations':asdict(perturbations),'validation_key':asdict(key)}
    write_json_atomic(config_path,config);write_json_atomic(output/'empty_eval_index.json',{'schema':'isg-instance-index-v1','n':0,'instances':[]})
    auth={'campaign_id':'campaign_20261110','purpose':'training_pipeline_canary','config_sha256':file_digest(config_path),
          'start':index,'count':1,'max_instances':1,'expires_at':(datetime.now(timezone.utc)+timedelta(hours=6)).isoformat(),
          'stage':'fresh_bound_native_demo_then_own_replay','source_authorization':{'path':row['authorization_path'],'sha256':row['authorization_sha256']},
          'native_proof':proof,'max_attempts':3,'sample_time_budget_seconds':1800,'production_released':False,'eligible_for_goal':False}
    authorization=output/'authorization.json';write_json_atomic(authorization,auth)
    entry={k:auth[k] for k in ('purpose','config_sha256','start','count','max_instances','expires_at')}
    entry.update(authorization_path=str(authorization),authorization_sha256=file_digest(authorization))
    env={**job['environment'],'ISPATIALGYM_CAMPAIGN_CONFIG':str(config_path),'TEMPLATE':bound.scene_setup['source_template'],
         'EVAL_INDEX':str(output/'empty_eval_index.json'),'START_PER_ATTEMPT':'0','MAX_ATTEMPTS':'3',
         'ISGF_COLLECT_EXTRA':'--no-goal-override','ISPATIALGYM_FAST_BAND_SHARE':'0','ISPATIALGYM_FAST_COLLECTOR_PREBUILD':'0',
         'ISGF_SAMPLING_CACHE':str(root/'cache'/token),'ISGF_ASSET_CACHE':str(root/'asset_cache'),
         'ISGF_JB_NFS':str(root/'jobs'),'ISGF_PREDEMO_ROOT':str(root),'KIT_TPL_NFS_ROOT':str(root/'kit_templates'),
         'KIT_TEX_STORE':str(root/'kit_textures'),'ISGF_EXP_PATH':str(root/'isaacsim_overlay/apps')}
    cmd=[sys.executable,str(native/'scripts/trainfast/submit_fast.py'),'submit',row['class_id'],'1','--flow','fast',
         '--code',str(native),'--root',str(root),'--ledger',str(root/'manifests'/('dispatch_'+token+'_allocations.tsv')),
         '--base-index',str(index),'--tag','training_height_annotated_r1','--job-prefix','isg261110_',
         '--grasps','v2only','--v2-root',job['dependency_release'],'--map-overlay','off','--alias-memory','off','--no-memory-aggregate',
         '--customs','template','--kcollect','1','--sample-limit-s','1800','--time-s','9000']
    for k,v in env.items():cmd+=['--extra-env',k+'='+v]
    register(root,entry)
    decision=check_environment(root,env,start=index,count=1);require(decision['allowed'] and decision['category']=='validation_canary','annotated canary gate failed')
    request={'campaign_id':'campaign_20261110','candidate_id':row['candidate_id'],'class_id':row['class_id'],
             'native_proof':proof,'config_path':str(config_path),'config_sha256':file_digest(config_path),
             'start':index,'count':1,'directory':str(output),'command':cmd,'environment':env,'authorization':entry,
             'status':'ready_for_guarded_gpu_submission','production_released':False,'eligible_for_goal':False,'quota_consumed':0}
    write_json_atomic(output/'request.json',request);return request


def observe(root,requests):
    root=Path(root);document=read(requests);require(document['campaign_id']=='campaign_20261110','foreign request batch')
    destination=Path(requests).parent/'stream';destination.mkdir(exist_ok=True)
    with atomic_directory_lock(destination/'state.lockdir'):
        state=read(destination/'state.json') if (destination/'state.json').exists() else {'campaign_id':'campaign_20261110','records':{}}
        from scripts.campaign_20261110.controller_families import allocations
        from scripts.campaign_20261110.training_canaries import publish_canary
        allocated=None
        for row in document['records']:
            previous=state['records'].get(row['candidate_id'],{})
            if previous.get('status')=='ready_for_guarded_gpu_submission':
                if previous.get('annotated_status')=='validated_canary_exported':continue
                if allocated is None:allocated=allocations(root)
                matches=[a for a in allocated if int(a['start'])==previous['start'] and a['key']==row['class_id']]
                if not matches:continue
                try:
                    require(len(matches)==1,'ambiguous annotated allocation')
                    allocation=matches[0];job=Path(allocation['job_dir'])
                    from scripts.campaign_20261110.production_gate import read_environment
                    env=read_environment(job/'job.env')
                    require(file_digest(env['ISPATIALGYM_CAMPAIGN_CONFIG'])==previous['config_sha256'],'annotated allocation config differs')
                    gate=root/'validation/open_loop'/row['class_id']/allocation['jobid']/'gate.json'
                    previous.update(allocation=allocation,annotated_status='awaiting_own_demo_and_replay')
                    if gate.exists():
                        require(read(gate).get('success') is True,'own native physics replay failed')
                        previous['validation_review']=publish_canary(job,gate,root=root,started_at=1794321600)
                        previous['annotated_status']='validated_canary_exported'
                    previous.pop('annotated_error',None)
                except Exception as exc:
                    previous.update(annotated_status='blocked',annotated_error=type(exc).__name__+': '+str(exc))
                continue
            try:
                prepared=prepare_annotated(row,root=root)
                value={'status':'native_pending'} if prepared is None else {'status':prepared['status'],'request':str(Path(row['config_path']).parent/('annotated_'+row['candidate_id'])/'request.json'),'start':prepared['start'],'config_sha256':prepared['config_sha256']}
            except Exception as exc:
                value={'status':'blocked','reason':type(exc).__name__+': '+str(exc)}
            if previous and previous!=value:state.setdefault('history',[]).append({'candidate_id':row['candidate_id'],'previous':previous,'at':time.time()})
            state['records'][row['candidate_id']]=value
        state['observed_at']=datetime.now(timezone.utc).isoformat();write_json_atomic(destination/'state.json',state)
    return state


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--requests',type=Path,required=True);p.add_argument('--watch',action='store_true');a=p.parse_args()
    while True:
        try:print(json.dumps(observe(a.root,a.requests)),flush=True)
        except Exception as exc:print(json.dumps({'error':type(exc).__name__+': '+str(exc)}),flush=True)
        if not a.watch:break
        time.sleep(30)
