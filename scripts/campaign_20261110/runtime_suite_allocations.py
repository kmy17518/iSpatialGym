"""Publish exact suite allocation and per-member replay evidence, without credit."""
from pathlib import Path
import argparse,csv,hashlib,json,time
from .runtime_validation_metrics import checked,reference,register,require,write,validate_row
from .training_diversity import observe_ticket_jobs,_terminal_jobs


def immutable(directory,value):
    data=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
    path=Path(directory)/(hashlib.sha256(data).hexdigest()+'.json');path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():require(path.read_bytes()==data,'immutable evidence conflict')
    else:
        with path.open('xb') as stream:stream.write(data)
    return reference(path)


def allocation(row,receipt,root):
    require(receipt['status']=='submitted','allocation not submitted')
    a=receipt['allocation']
    require(receipt['token']==row['dispatch_token'],'dispatch token differs')
    require((int(a['start']),int(a['n']))==(row['fresh_start'],row['fresh_count']),'allocation range differs')
    require(Path(a['code']).resolve()==Path(row['frozen_runtime']['path']).resolve(),'allocation runtime differs')
    require(a['key']==row['class_id'],'allocation class differs')
    require(Path(a['job_dir']).resolve().is_relative_to(root/'out'),'source outside campaign')
    ledger=Path(a['ledger']).resolve();require(ledger.is_relative_to(root/'manifests'),'ledger outside campaign')
    with ledger.open() as stream:matches=[r for r in csv.DictReader(stream,delimiter='\t') if r['jobid']==str(a['jobid'])]
    require(len(matches)==1 and all(str(matches[0][k])==str(a[k]) for k in ('job_dir','code','start','n','key')),'ledger allocation differs')
    return a


def gate_member(path,row,source):
    gate=json.loads(Path(path).read_text())
    require(Path(gate['source_job']).resolve()==source.resolve(),'member source differs')
    require(Path(gate['snapshot']).resolve()==Path(row['frozen_runtime']['path']).resolve(),'member runtime differs')
    demo=Path(gate['demo']);tro=Path(gate['instance'])
    require(demo.resolve().is_relative_to(source.resolve()/'demo') and tro.resolve().is_relative_to(source.resolve()/'inst'),'member inputs escape source')
    require(reference(demo)['sha256']==gate['demo_sha256'] and reference(tro)['sha256']==gate['tro_sha256'],'member inputs changed')
    require(tro.name==demo.parent.name+'-tro_state.json','member identity differs')
    return demo.parent.name,gate


def observe(root,run_dir):
    root=Path(root).resolve();run_dir=Path(run_dir);start=time.time();rows=[];errors=[];all_jobs={}
    central_path=root/'validation/controller.json'
    central=json.loads(central_path.read_text()).get('gates',{}) if central_path.exists() else {}
    for p in sorted((run_dir/'rows').glob('*.json')):
        try:
            row=json.loads(p.read_text());validate_row(row);register(p)
            rpath=root/'dispatch/receipts'/(row['dispatch_token']+'.json')
            if not rpath.exists():continue
            receipt=json.loads(rpath.read_text())
            if receipt.get('status')!='submitted':continue
            a=allocation(row,receipt,root);source=Path(a['job_dir']);jid=str(a['jobid']);backend='vast' if jid.startswith('9000') else 'cluster'
            value={'schema':'isg-current-runtime-class-allocation-v1','campaign_id':root.name,'row':reference(p),'allocation':a,
                   'dispatch_receipt':immutable(run_dir/'evidence/dispatch_receipts',receipt),'records':[], 'member_gates':[]}
            value['records'].append({'job_id':jid,'job_dir':str(source),'backend':backend,'stage':'generation_collection','submission_ref':value['dispatch_receipt']})
            entries=[];c=central.get(jid)
            if c:entries.append(c)
            member_path=run_dir/'member_replays'/p.name
            if member_path.exists():
                state=json.loads(member_path.read_text());require(state['row']==reference(p),'member owner row differs');entries.extend(state.get('members',{}).values())
            gates={};pending=[]
            for e in entries:
                replay_id=e.get('replay_job_id')
                if replay_id:
                    entry_ref=immutable(run_dir/'evidence/replay_ownership',e)
                    rb='vast' if str(replay_id).startswith('9000') else 'cluster'
                    value['records'].append({'job_id':str(replay_id),'job_dir':e['output'],'backend':rb,'stage':'physics_replay','submission_ref':entry_ref})
                    for old in e.get('scheduler_attempt_history',[]):
                        value['records'].append({'job_id':str(old['replay_job_id']),'job_dir':e['output'],'backend':'cluster','stage':'physics_replay','submission_ref':entry_ref})
                gp=Path(e.get('output','/nonexistent'))/'gate.json'
                if gp.is_file():
                    member,g=gate_member(gp,row,source)
                    if g.get('success') is True and g.get('replay_rc')==0:gates[member]=reference(gp)
                elif replay_id:pending.append(str(replay_id))
            value['records']=list({(r['backend'],r['job_id']):r for r in value['records']}.values())
            value['member_gates']=[gates[k] for k in sorted(gates)]
            canonical=Path(c['output'])/'gate.json' if c and c.get('output') else root/'validation/open_loop'/a['key']/jid/'gate.json'
            if canonical.is_file():
                _,g=gate_member(canonical,row,source)
                if g.get('success'):value['gate']=reference(canonical)
            if 'gate' not in value and value['member_gates']:value['gate']=value['member_gates'][0]
            jobs=[{'job_id':j['job_id'],'backend':j['backend'],'stage':'collection' if j['stage']=='generation_collection' else 'replay'} for j in value['records']]
            for j in jobs:all_jobs[(j['backend'],j['job_id'],j['stage'])]=j
            target=run_dir/'allocations'/p.name;prior=json.loads(target.read_text()) if target.exists() else {}
            if prior.get('terminal_sealed') and all(prior.get(k)==value.get(k) for k in ('row','allocation','dispatch_receipt','records','member_gates','gate')):
                rows.append((target,None,None));continue
            rows.append((target,value,jobs))
        except Exception as exc:errors.append({'row':str(p),'error':repr(exc)})
    changing=[r for r in rows if r[1] is not None]
    terminal=None
    if changing:
        controller=immutable(run_dir/'evidence/job_inventories',{'schema':'isg-training-ticket-controller-v1','campaign_id':root.name,'owner':'compute_runtime_suite','jobs':list(all_jobs.values())})
        terminal_ref=observe_ticket_jobs(root,controller,run_dir/'evidence/backend_observations'/str(time.time_ns()))
        terminal=checked(terminal_ref)
    for target,value,jobs in changing:
        keys={(r['backend'],r['job_id'],r['stage']) for r in jobs}
        selected={**terminal,'jobs':[j for j in terminal['jobs'] if (j['backend'],j['job_id'],j['stage']) in keys]}
        value['terminal_evidence']=immutable(run_dir/'evidence/class_backend_observations',selected)
        value['terminal_sealed']=False
        try:
            _terminal_jobs(root,{'jobs':jobs},selected,selected['observed_at'])
            row=checked(value['row']);source=Path(value['allocation']['job_dir'])
            successes={m.parent.name for m in (source/'demo').glob('*/meta.json') if json.loads(m.read_text()).get('success') is True}
            all_terminal=set()
            # Gate failures remain terminal evidence but never pass member_gates.
            central_entry=central.get(str(value['allocation']['jobid']),{})
            statepath=run_dir/'member_replays'/target.name
            entries=[central_entry]+(list(json.loads(statepath.read_text()).get('members',{}).values()) if statepath.exists() else [])
            for e in entries:
                gp=Path(e.get('output','/nonexistent'))/'gate.json'
                if gp.exists():all_terminal.add(gate_member(gp,row,source)[0])
            value['terminal_sealed']=(source/'sampling.done').exists() and bool(successes) and successes<=all_terminal
        except (ValueError,OSError,KeyError):pass
        write(target,value)
    summary={'observed_at':time.time(),'elapsed_seconds':time.time()-start,'allocations':len(rows),'updated':len(changing),'errors':errors,'owner':'compute_runtime_suite','acceptance_authority':False}
    write(run_dir/'allocation_observer.json',summary);return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--every',type=float,default=20);a=p.parse_args()
    from .storage import atomic_directory_lock
    with atomic_directory_lock(a.run_dir/'allocation_observer.lockdir'):
        while True:
            try:print(json.dumps(observe(a.root,a.run_dir)),flush=True)
            except Exception as e:print(json.dumps({'error':repr(e),'observed_at':time.time()}),flush=True)
            if not a.every:return
            time.sleep(a.every)
if __name__=='__main__':main()
