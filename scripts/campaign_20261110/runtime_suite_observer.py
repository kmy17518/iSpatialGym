"""One bounded acceptance owner for a current-runtime suite; never dispatches."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from .contracts import digest, file_digest, require, write_json_atomic
from .runtime_suite import active, checked, registered_rows
from .runtime_suite_acceptance import reconcile, reference, immutable
from .storage import atomic_directory_lock


def source_stage(root,row,wrapper):
    """Submission alone proves no worker, native instance, or demo progress."""
    from .acceptance import report_passed
    from .runtime_suite import timestamp
    from .training_diversity import _terminal_jobs
    job=Path(wrapper['allocation']['job_dir']);since=timestamp(row['run_started_at'])
    native=list((job/'inst').rglob('*-tro_state.json')) if (job/'inst').exists() else []
    successful=[];failed=[]
    for path in (job/'demo').glob('*/meta.json'):
        try:
            value=json.loads(path.read_text());report_passed(value,'oracle')
            require(timestamp(value['started_utc'])>=since and
                    Path(value['provenance']['isg_root']).resolve()==Path(row['frozen_runtime']['path']).resolve(),
                    'demo source runtime/time differs')
            successful.append(reference(path))
        except (OSError,ValueError,KeyError,TypeError):failed.append(str(path))
    worker=any(path.exists() and path.stat().st_size>0 and path.stat().st_mtime>=since
               for path in (job/'job.txt',job/'sample.log'))
    terminal=False
    if wrapper.get('terminal_evidence'):
        proof=checked(root,wrapper['terminal_evidence'])
        jobs=[j for j in proof['jobs'] if j['job_id']==str(wrapper['allocation']['jobid']) and j['stage'] in ('native','collection')]
        require(len(jobs)==1,
                'source terminal proof refers to another allocation')
        try:_terminal_jobs(root,{'jobs':jobs},{**proof,'jobs':jobs},proof['observed_at']);terminal=True
        except (OSError,ValueError,KeyError,TypeError):pass
    needed=1 if row['camera_exempt'] else 2
    if len(successful)>=needed:status='awaiting_own_replay'
    elif terminal:
        status=('native_not_started_terminal' if not worker and not native else
                'native_failed' if not native else 'demo_failed' if not successful else 'insufficient_successful_demos')
    elif successful:status='awaiting_distinct_demos'
    elif native:status='awaiting_demo'
    elif worker:status='native_started'
    else:status='awaiting_worker'
    return {'status':status,'native_written_observed':len(native),'successful_demo_reports':successful,
            'failed_demo_report_paths':failed,'worker_start_evidence_observed':worker,'source_terminal_verified':terminal}


def observe(root, run_dir):
    root=Path(root).resolve();base=Path(run_dir).resolve()
    pointer,release,manifest,_,current=active(root)
    require(base==current,'acceptance observer is not the active immutable suite')
    state_path=base/'acceptance_observer.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'schema':'isg-runtime-suite-acceptance-observer-v1','classes':{}}
    with atomic_directory_lock(base/'acceptance_observer.lockdir'):
        state.update(run_id=release['run_id'],cycle_started_at=time.time(),owner='/root/dataset_contract')
        # A known completed dependency must not wait behind a full cohort's
        # still-starting native directories on NFS.
        def priority(entry):
            prior=state['classes'].get(entry['class_id'],{})
            return (0 if prior.get('gate') and not prior.get('validation_passed') else
                    1 if prior.get('status') in ('awaiting_own_replay','awaiting_member_replay','awaiting_terminal_evidence') else 2,
                    entry['class_id'])
        for entry in sorted(manifest['classes'],key=priority):
            cls=entry['class_id'];row_path=base/'rows'/(cls+'.json');allocation_path=base/'allocations'/(cls+'.json')
            status={'observed_at':time.time(),'status':'awaiting_authored_row','validation_passed':False}
            try:
                if not row_path.exists():
                    state['classes'][cls]=status;continue
                row_ref=registered_rows(root,pointer,base,cls)[-1];row=checked(root,row_ref)
                if row.get('attempt_number'):
                    allocation_path=Path(row_ref['path']).parent/'allocation.json'
                status.update(row=row_ref,status='awaiting_allocation')
                if not allocation_path.exists():
                    state['classes'][cls]=status;continue
                # The publisher updates its pointer while physical re-audit may
                # take minutes. Pin one complete atomic version before audit.
                allocation=json.loads(allocation_path.read_text())
                allocation_ref=immutable(base/'evidence/acceptance_inputs'/(digest(allocation)+'.json'),allocation)
                status.update(allocation=allocation_ref,status='awaiting_native_and_demo')
                if not allocation.get('allocation'):
                    state['classes'][cls]=status;continue
                receipt=checked(root,allocation['dispatch_receipt']);job=allocation['allocation']
                gate_path=(Path(allocation['gate']['path']) if allocation.get('gate') else
                    root/'validation/open_loop'/receipt['allocation']['key']/str(job['jobid'])/'gate.json')
                if not gate_path.exists():
                    status.update(source_stage(root,row,allocation));state['classes'][cls]=status;continue
                gate_ref=reference(gate_path);gate=checked(root,gate_ref)
                status['gate']=gate_ref
                if gate.get('success') is not True:
                    status.update(status='replay_failed',reason=gate.get('reasons',gate.get('result')))
                    state['classes'][cls]=status;continue
                if not row['camera_exempt'] and len(allocation.get('member_gates',[]))<2:
                    status['status']='awaiting_member_replay';state['classes'][cls]=status;continue
                if not allocation.get('terminal_evidence'):
                    status['status']='awaiting_terminal_evidence';state['classes'][cls]=status;continue
                signature=digest({'row':row_ref,'allocation':allocation_ref,'gate':gate_ref})
                prior=state['classes'].get(cls,{})
                if prior.get('input_signature')==signature and prior.get('validation_passed') is True:
                    state['classes'][cls]=prior;continue
                state['current']={'class_id':cls,'status':'strict_reaudit','started_at':time.time()}
                write_json_atomic(state_path,state)
                result=reconcile(root,row_ref,allocation_ref,gate_ref)
                status.update(status=result['status'],validation_passed=result['validation_passed'],
                    result=reference(base/'acceptance'/(cls+'.json')),input_signature=signature)
            except FileExistsError as error:
                status.update(status='awaiting_acceptance_lock',error=repr(error),
                    next_action='Retry normal coherent validation after its live lock releases; preserve source and all member replay proofs.')
            except (OSError,ValueError,KeyError,TypeError) as error:
                status.update(status='audit_blocked',error=repr(error),next_action='Inspect exact missing/failed binding; preserve original source and do not resample unchanged.')
            state['classes'][cls]=status
            write_json_atomic(state_path,state)
        state.update(cycle_finished_at=time.time(),current=None)
        write_json_atomic(state_path,state)
    return state


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--run-dir',required=True)
    p.add_argument('--interval',type=float,default=60);p.add_argument('--once',action='store_true')
    args=p.parse_args()
    while True:
        try:
            state=observe(args.root,args.run_dir)
            print(json.dumps({'observed_at':time.time(),'classes':len(state['classes']),
                'validated':sum(r.get('validation_passed') is True for r in state['classes'].values())}),flush=True)
        except (OSError,ValueError,KeyError,TypeError) as error:
            print(json.dumps({'observed_at':time.time(),'error':repr(error)}),flush=True)
        if args.once:return
        time.sleep(max(5,args.interval))


if __name__=='__main__':main()
