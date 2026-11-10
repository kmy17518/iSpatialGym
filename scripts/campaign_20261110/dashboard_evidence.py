"""Read-only allocator evidence for the campaign dashboard; never submits jobs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import time

from scripts.campaign_20261110.contracts import digest, file_digest, require, task_quotas, task_for_alias, write_json_atomic
from scripts.campaign_20261110.export_release import snapshot_allocations, template_from
from scripts.campaign_20261110.group_lane import physical_template, conceptual_template
from scripts.campaign_20261110.storage import atomic_directory_lock

SPLITS = ("eval_standard","eval_model","eval_category","train")


def read(path):
    return json.loads(Path(path).read_text())


def timestamp(value):
    return datetime.datetime.fromtimestamp(value,datetime.timezone.utc).isoformat()


def source_record(path, kind):
    path=Path(path)
    return {"kind":kind,"path":str(path),"sha256":file_digest(path),"mtime":path.stat().st_mtime}


def observed_json(path, kind):
    """Hash exactly the bytes used for an observation, even during atomic writes."""
    path=Path(path)
    with path.open('rb') as handle:
        import os
        stat=os.fstat(handle.fileno());data=handle.read()
    return json.loads(data),{'kind':kind,'path':str(path),'sha256':hashlib.sha256(data).hexdigest(),'mtime':stat.st_mtime}


def epoch(value, fallback):
    if isinstance(value,(int,float)):return value
    try:return datetime.datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()
    except (ValueError,TypeError,AttributeError):return fallback


def config_scope(path, root, sources):
    path=Path(path).resolve();require(path.is_relative_to(Path(root).resolve()),'reason config outside campaign')
    raw,source=observed_json(path,'reason_campaign_config');sources.append(source)
    template=template_from(raw['template'])
    require(template.fingerprint==raw['validation_key']['template_sha256'] and template.split==raw['split'],
            'reason config identity differs')
    return {'task':template.task,'class_id':template.class_id,'split':template.split}


def job_config(job,root):
    job=Path(job).resolve();require(job.is_relative_to(Path(root).resolve()/'out'),'reason job outside campaign output')
    for line in (job/'job.env').read_text().splitlines():
        if line.startswith('ISPATIALGYM_CAMPAIGN_CONFIG='):
            values=shlex.split(line.split('=',1)[1]);require(len(values)==1,'ambiguous reason config')
            return values[0]
    return None


def standard_catalog(path):
    raw=read(path);rows=raw["templates"] if isinstance(raw,dict) else raw
    result={}
    for row in rows:
        name=row.get("class_id",row.get("key"))
        task=row.get("task") or task_for_alias(name)
        require(name and name not in result,"standard inventory has duplicate or missing class id")
        result[name]=task
    quotas={q.task:q for q in task_quotas()}
    counts=Counter(result.values())
    require(len(result)==100 and counts=={task:q.standard_classes for task,q in quotas.items()},
            "dashboard requires the complete frozen 100-class/23-task inventory")
    return result


def controller_reasons(root, catalog):
    reasons=defaultdict(list);sources=[];errors=[]
    for directory in sorted((Path(root)/"controllers").iterdir()) if (Path(root)/"controllers").exists() else []:
        config_path,state_path=directory/"config.json",directory/"state.json"
        if not state_path.is_file():
            continue
        try:
            state,source=observed_json(state_path,'controller_state');sources.append(source)
            config={}
            if config_path.exists():
                config,config_source=observed_json(config_path,'controller_config');sources.append(config_source)
            seeds=config.get('seeds',config.get('families',[]))
            if isinstance(seeds,dict):
                seeds=[dict(value,id=key) for key,value in seeds.items()]
            if config.get('seed_catalog') and config.get('schema')=='isg-group-controller-v1':
                path=Path(config['seed_catalog']).resolve()
                require(path.is_relative_to(Path(root).resolve()/'controllers'),'controller seed catalog outside campaign')
                admitted,catalog_source=observed_json(path,'admitted_seed_catalog');sources.append(catalog_source)
                require(admitted.get('schema')=='isg-group-seed-catalog-v1'
                        and admitted.get('campaign_id')==config.get('campaign_id'),'controller seed catalog identity differs')
                seeds=list(seeds)+admitted['seeds']
            if not seeds and state.get('families'):
                seeds=[{'id':key,'class_id':key} for key in state['families']]
            for spec in seeds:
                class_id=spec.get("class_id")
                if not class_id:
                    continue
                task=catalog.get(class_id)
                if task is None:
                    try:task=task_for_alias(class_id)
                    except ValueError:continue
                current=state.get('seeds',state.get('families',{})).get(spec.get('id',class_id),{})
                scope={}
                candidate=spec.get('seed_config')
                if not candidate:
                    seed_directory=spec.get('inputs',spec.get('seed'))
                    current_stage=current.get('current') or {}
                    seed_directory=seed_directory or (current_stage.get('group') or current_stage.get('anchor') or {}).get('directory')
                    if seed_directory:candidate=str(Path(seed_directory)/'campaign_config.json')
                if candidate and Path(candidate).exists():
                    scope=config_scope(candidate,root,sources)
                    require(scope['class_id']==class_id,'controller reason class differs from config')
                status=current.get("status") or current.get("phase") or "status not recorded"
                message=current.get("error") or current.get("runtime_failure",{}).get("reason") or status
                reasons[(task,class_id)].append({"controller":directory.name,"status":status,"reason":str(message),
                    'split':scope.get('split',spec.get('split')),'observed_at':epoch(state.get('updated_at',state.get('updated_at_unix')),source['mtime']),
                    "state_source":str(state_path),"source_kind":"controller_observation"})
            for identity,record in state.get('records',{}).items():
                if not isinstance(record,dict) or not record.get('admission_id'):
                    continue
                scope=config_scope(record['prepared']['config'],root,sources)
                status=record.get('status','status not recorded')
                message=record.get('error') or record.get('reason') or {
                    'ready':'Fresh training canary prepared; awaiting dispatch.',
                    'external_submission':'Fresh training canary queued or running; exact runtime demo and replay gate required.',
                    'admitted':'Exact runtime training gate passed; seed admitted to bounded refill.',
                }.get(status,status)
                reasons[(scope['task'],scope['class_id'])].append({
                    'controller':directory.name,'admission_id':identity,'status':status,'reason':str(message),
                    'split':scope['split'],'observed_at':epoch(state.get('updated_at'),source['mtime']),
                    'state_source':str(state_path),'source_kind':'controller_observation'})
        except (ValueError,KeyError,TypeError,OSError) as exc:
            errors.append({"source":str(directory),"error":type(exc).__name__+": "+str(exc)})
    return reasons,sources,errors


def failure_observation(job, root):
    """Summarize structured failures; raw logs never enter dashboard responses."""
    job=Path(job).resolve();root=Path(root).resolve()
    require(job.is_relative_to(root/'out'),'failure source outside campaign')
    rejected=Counter();failed=Counter();proof=[];parts=[]
    for path in sorted((job/'inst').glob('*/*/template_meta.json')):
        data,source=observed_json(path,'sampling_failure_summary');proof.append(source)
        rejected.update({k:v for k,v in data.get('rejections',{}).items() if isinstance(v,int) and v>0})
    path=job/'results.jsonl'
    if path.is_file():
        body=path.read_bytes();proof.append({'path':str(path),'sha256':hashlib.sha256(body).hexdigest(),
                                            'mtime':path.stat().st_mtime,'kind':'demo_failure_summary'})
        for line in body.decode(errors='replace').splitlines():
            try:row=json.loads(line)
            except ValueError:continue
            if row.get('success') is False:failed[str(row.get('reason') or row.get('status') or 'unspecified')]+=1
    if failed:parts.append('demo failures: '+', '.join(f'{name} ({count})' for name,count in failed.most_common(3)))
    if rejected:parts.append('sampling rejections: '+', '.join(f'{name} ({count})' for name,count in rejected.most_common(3)))
    if not parts and (job/'sample.rc').exists():
        code=(job/'sample.rc').read_text().strip()
        if code and code!='0':parts.append('sampler exit '+code+'; no structured rejection report available')
    return '; '.join(parts),proof


def validation_reasons(root, catalog):
    reasons=defaultdict(list);sources=[];errors=[]
    path=Path(root)/'validation/controller.json'
    if not path.exists():return reasons,sources,errors
    try:
        state,source=observed_json(path,'validation_controller');sources.append(source)
        for job_id,row in state.get('gates',{}).items():
            class_id=row.get('key');task=catalog.get(class_id)
            split=None
            progress=[]
            if row.get('source_job_dir'):
                try:
                    job=Path(row['source_job_dir']).resolve()
                    require(job.is_relative_to(Path(root).resolve()/'out'),'validation job outside campaign')
                    if (job/'job.env').exists():progress.append((job/'job.env').stat().st_mtime)
                    path_config=job_config(row['source_job_dir'],root)
                    if path_config:
                        scope=config_scope(path_config,root,sources)
                        class_id,task,split=scope['class_id'],scope['task'],scope['split']
                except (ValueError,KeyError,TypeError,OSError) as exc:
                    errors.append({'source':row['source_job_dir'],'error':'validation reason scope: '+str(exc)})
            if task is None:
                try:task=task_for_alias(class_id or '')
                except ValueError:continue
            output=Path(row['output']).resolve() if row.get('output') else None
            if output and output.is_relative_to(Path(root).resolve()/'validation') and output.exists():
                progress.append(output.stat().st_mtime)
            validated_at=epoch((row.get('gate') or {}).get('validated_at'),0)
            if validated_at:progress.append(validated_at)
            message=str(row.get('reason') or row.get('error') or row.get('status'))
            failure_sources=[]
            if row.get('source_job_dir') and row.get('status') in ('demo_missing','demo_failed','replay_failed'):
                try:detail,failure_sources=failure_observation(row['source_job_dir'],root)
                except (ValueError,OSError,TypeError) as exc:
                    detail='structured failure observation unavailable'
                    errors.append({'source':row['source_job_dir'],'error':'failure summary: '+str(exc)})
                if detail and not (row.get('reason') or row.get('error')):message+='; '+detail
                if row.get('status')=='replay_failed':
                    result=(row.get('gate') or {}).get('result') or {}
                    if result.get('reason') or result.get('error'):
                        message+='; replay: '+str(result.get('reason') or result.get('error'))
            reasons[(task,class_id)].append({'source_kind':'validation_observation','job_id':job_id,
                'state_source':str(path),'source_job_dir':row.get('source_job_dir'),'split':split,
                'status':row.get('status'),'reason':message,'failure_sources':failure_sources,
                'observed_at':epoch(row.get('updated_at',state.get('updated_at')),source['mtime']),
                'progress_at':max(progress) if progress else None})
    except (ValueError,KeyError,TypeError,OSError) as exc:
        errors.append({'source':str(path),'error':type(exc).__name__+': '+str(exc)})
    return reasons,sources,errors


def pipeline_reasons(root, catalog):
    """Attach hash-verified preparation evidence without promoting any counts."""
    root=Path(root).resolve();path=root/'manifests/pipeline_status.json'
    reasons=defaultdict(list);sources=[];errors=[];verified={}
    if not path.exists():return reasons,sources,errors
    try:
        data,source=observed_json(path,'pipeline_status');sources.append(source)
        require(data.get('schema')=='isg-campaign-pipeline-status-v1' and data.get('campaign_id')=='campaign_20261110',
                'pipeline observation belongs to another schema or campaign')
        tasks={q.task for q in task_quotas()}
        for ordinal,row in enumerate(data['rows']):
            try:
                task,class_id,split=row['task'],row['class_id'],row['split']
                require(task in tasks and split in SPLITS,'unknown pipeline task or split')
                require(split=='eval_category' or catalog.get(class_id)==task,'pipeline class differs from standard inventory')
                require(row.get('reason') and row.get('stage') and row.get('evidence'),'pipeline observation lacks evidence')
                for item in row['evidence']:
                    evidence=Path(item['path']).resolve()
                    require(evidence.is_relative_to(root) and evidence.is_file(),'pipeline evidence outside campaign or unavailable')
                    if evidence not in verified:
                        verified[evidence]=source_record(evidence,'pipeline_evidence');sources.append(verified[evidence])
                    require(verified[evidence]['sha256']==item['sha256'],'pipeline evidence changed after observation')
                reasons[(task,class_id)].append({'source_kind':'pipeline_observation','split':split,
                    'status':row['stage'],'reason':row['reason'],'observed_at':epoch(row.get('observed_at'),source['mtime']),
                    'state_source':str(path),'evidence':row['evidence']})
            except (ValueError,KeyError,TypeError,OSError) as exc:
                errors.append({'source':str(path),'row':ordinal,'error':type(exc).__name__+': '+str(exc)})
    except (ValueError,KeyError,TypeError,OSError) as exc:
        errors.append({'source':str(path),'error':type(exc).__name__+': '+str(exc)})
    return reasons,sources,errors


def _identities(row, root, cache):
    physical,conceptual=row.get("physical_template_sha256"),row.get("conceptual_template_sha256")
    if physical and conceptual:
        return physical,conceptual
    key=row["validation_key"]["template_sha256"]
    if key not in cache:
        if row.get("contract"):
            raw=row["contract"]
        else:
            group=root/"groups"/row["group_id"]/"campaign_config.json"
            require(group.resolve().is_relative_to(root/"groups"),"invalid accepted group path")
            raw=read(group)["template"]
        template=template_from(raw)
        require(template.fingerprint==key,"accepted identity contract no longer matches its validation key")
        cache[key]=(physical_template(template),conceptual_template(template))
    return cache[key]


def coverage_reason(evidence):
    """Keep old failed candidates from masking a later validated configuration."""
    if not evidence:return 'No controller evidence in this snapshot.'
    controller=[row for row in evidence if row.get('source_kind')=='controller_observation']
    if controller:
        return max(controller,key=lambda row:row.get('observed_at') or 0)['reason']
    validation=[row for row in evidence if row.get('source_kind')=='validation_observation']
    if validation and any(row.get('status')=='validated' for row in validation):
        latest=max(validation,key=lambda row:row.get('progress_at') or row.get('observed_at') or 0)
        if latest.get('status')=='validated':
            return 'Demo and physics replay passed; awaiting production or committed acceptance.'
        return ('At least one configuration passed demo and physics replay; latest candidate observation: '
                +latest['reason']+'. Remaining instances are not yet accepted.')
    latest=max(evidence,key=lambda row:row.get('progress_at') or row.get('observed_at') or 0)
    return latest['reason']


def summarize(rows, catalog, *, root, observed_at, reasons=None, sources=(), errors=()):
    """Count committed IDs only; geometry candidates and demo logs are not input."""
    root=Path(root).resolve();reasons=reasons or {}
    normalized=[]
    for row in rows:
        if not row.get('class_id'):
            template=template_from(row['contract'])
            require(template.fingerprint==row['validation_key']['template_sha256'] and template.task==row['task'],
                    'legacy camera class contract differs from its accepted key')
            row={**row,'class_id':template.class_id}
        normalized.append(row)
    rows=normalized
    require(len({row['instance_id'] for row in rows})==len(rows),"instance appears in multiple authoritative allocators")
    quotas={q.task:q for q in task_quotas()}
    require(all(row['task'] in quotas and row['split'] in SPLITS for row in rows),"unknown accepted task/split")
    counts=Counter(row['split'] for row in rows)
    by_class=Counter((row['task'],row['class_id'],row['split']) for row in rows)
    by_task=Counter((row['task'],row['split']) for row in rows)
    class_rows=[]
    def append(task,class_id,split,target,accepted,scope):
        evidence=[row for row in reasons.get((task,class_id),()) if row.get('split')==split]
        if split=='eval_category':
            evidence=[entry for (t,_),entries in reasons.items() if t==task for entry in entries if entry.get('split')==split]
        reason=coverage_reason(evidence)
        complete=target is not None and accepted>=target
        class_rows.append({'class_id':class_id,'task':task,'split':split,'accepted':accepted,'target':target,
            'remaining':max(0,target-accepted) if target is not None else None,'scope':scope,
            'reason':'target reached' if complete else reason,
            'reason_evidence':evidence})
    for class_id,task in sorted(catalog.items()):
        q=quotas[task]
        append(task,class_id,'eval_standard',10,by_class[task,class_id,'eval_standard'],'standard_class')
        if q.model_classes:
            append(task,class_id,'eval_model',q.model_instances_per_class,by_class[task,class_id,'eval_model'],'model_holdout_class')
        append(task,class_id,'train',q.train_instances//q.standard_classes,by_class[task,class_id,'train'],'standard_training_class')
    for task,q in sorted(quotas.items()):
        append(task,task,'eval_category',q.category_instances,by_task[task,'eval_category'],'category_holdout_task')
    for task,class_id,split in sorted(by_class):
        if split=='train' and class_id not in catalog:
            append(task,class_id,split,None,by_class[task,class_id,split],'additional_training_class_task_quota_applies')
    tasks=[]
    for task,q in sorted(quotas.items()):
        targets={'eval_standard':q.standard_classes*10,'eval_model':q.model_classes*q.model_instances_per_class,
                 'eval_category':q.category_instances,'train':q.train_instances}
        accepted={split:by_task[task,split] for split in SPLITS}
        tasks.append({'task':task,'counts':accepted,'targets':targets,
            'remaining':{split:max(0,targets[split]-accepted[split]) for split in SPLITS},
            'standard_classes':q.standard_classes,'model_classes':q.model_classes,'category_tasks':1})
    cache={};physical=set();conceptual=set();identity_errors=list(errors)
    for row in rows:
        try:
            p,c=_identities(row,root,cache);physical.add(p);conceptual.add(c)
        except (ValueError,KeyError,TypeError,OSError) as exc:
            identity_errors.append({'instance_id':row['instance_id'],'error':'template identity: '+str(exc)})
    return {'schema':'isg-dashboard-evidence-v1','campaign_id':'campaign_20261110',
        'observed_at':timestamp(observed_at),'observed_at_epoch':observed_at,
        'count_semantics':'Committed allocator IDs only; this read does not substitute for immutable release re-audit.',
        'counts':{split:counts[split] for split in SPLITS},'accepted_total':len(rows),
        'targets':{'eval_standard':1000,'eval_model':891,'eval_category':109,'train':100000},
        'denominators':{'standard_classes':100,'model_holdout_classes':99,'category_holdout_tasks':23,'tasks':23},
        'classes':class_rows,'tasks':tasks,'templates':{'accepted_physical':len(physical),'accepted_conceptual':len(conceptual),
            'accepted_classes':len({(row['task'],row['class_id']) for row in rows}),
            'accepted_contract_fingerprints':len({row['validation_key']['template_sha256'] for row in rows}),
            'accepted_validation_keys':len({digest(row['validation_key']) for row in rows}),
            'physical_definition':'Task, scene/rooms, bound object roles/models/asset hashes/default scales; excludes split, code and provenance.',
            'configuration_definition':'Exact validation keys additionally distinguish code, physics, data and sampled scale state.',
            'identity_audit_complete':not any('template identity:' in row.get('error','') for row in identity_errors)},
        'sources':list(sources),'errors':identity_errors,
        'unassigned_reason_evidence':[{'task':task,'class_id':class_id,**entry} for (task,class_id),entries in reasons.items()
                                      for entry in entries if entry.get('split') is None]}


def accepted_throughput(history, now):
    result=[]
    if not history:return result
    latest=history[-1]
    for window in (900,3600,10800):
        eligible=[row for row in history if now-window <= row['observed_at_epoch'] <= now]
        first=eligible[0] if eligible else latest
        elapsed=latest['observed_at_epoch']-first['observed_at_epoch']
        delta={split:latest['counts'][split]-first['counts'][split] for split in SPLITS}
        require(all(value>=0 for value in delta.values()),'accepted counter regressed; throughput unavailable')
        result.append({'window_seconds':window,'measured_seconds':elapsed,'accepted_delta':delta,
            'accepted_per_hour':{split:value*3600/elapsed for split,value in delta.items()} if elapsed>=60 else None,
            'total_accepted_per_hour':sum(delta.values())*3600/elapsed if elapsed>=60 else None})
        task_rows=[]
        for task,counts in latest.get('tasks',{}).items():
            if task not in first.get('tasks',{}):continue
            change={split:counts[split]-first['tasks'][task][split] for split in SPLITS}
            require(all(value>=0 for value in change.values()),'task accepted counter regressed')
            task_rows.append({'task':task,'accepted_delta':change,
                'accepted_per_hour':{split:value*3600/elapsed for split,value in change.items()} if elapsed>=60 else None})
        result[-1]['tasks']=task_rows
    return result


def build(root, standards, *, database=None, observed_at=None):
    root=Path(root).resolve();now=time.time() if observed_at is None else observed_at
    catalog=standard_catalog(standards)
    rows,_,sources=snapshot_allocations(root,database)
    reasons,controller_sources,errors=controller_reasons(root,catalog)
    validation,validation_sources,validation_errors=validation_reasons(root,catalog)
    for key,entries in validation.items():reasons[key].extend(entries)
    pipeline,pipeline_sources,pipeline_errors=pipeline_reasons(root,catalog)
    for key,entries in pipeline.items():reasons[key].extend(entries)
    return summarize(rows,catalog,root=root,observed_at=now,reasons=reasons,
        sources=[*sources,source_record(standards,'complete_standard_inventory'),*controller_sources,*validation_sources,*pipeline_sources],
        errors=errors+validation_errors+pipeline_errors)


def publish(report, output):
    output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    history_path=output.with_name('dashboard_accepted_history.json')
    with atomic_directory_lock(output.with_suffix('.lockdir')):
        history=read(history_path) if history_path.exists() else []
        sample={'observed_at_epoch':report['observed_at_epoch'],'counts':report['counts'],
                'tasks':{row['task']:row['counts'] for row in report['tasks']},
                'source_states_sha256':digest(report['sources'])}
        require(not history or sample['observed_at_epoch']>history[-1]['observed_at_epoch'],'dashboard sample time did not advance')
        history=(history+[sample])[-2880:]
        report['throughput']=accepted_throughput(history,report['observed_at_epoch'])
        report['accepted_history']=history[-360:]
        write_json_atomic(history_path,history)
        report['sources'].append(source_record(history_path,'accepted_count_history'))
        write_json_atomic(output,report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','standards','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--database');parser.add_argument('--interval',type=float,default=0)
    args=parser.parse_args();require(0<=args.interval<=60,'dashboard interval must be0..60 seconds')
    while True:
        try:
            report=publish(build(args.root,args.standards,database=args.database),args.output)
            print(json.dumps({'observed_at':report['observed_at'],'accepted':report['accepted_total'],'errors':len(report['errors'])}),flush=True)
        except Exception as exc:
            if args.interval==0:raise
            error={'observed_at':timestamp(time.time()),'error':type(exc).__name__+': '+str(exc),'previous_evidence_preserved':True}
            write_json_atomic(Path(args.output).with_name('dashboard_evidence_error.json'),error)
            print(json.dumps(error),flush=True)
        if args.interval==0:break
        time.sleep(args.interval)


if __name__=='__main__':main()
