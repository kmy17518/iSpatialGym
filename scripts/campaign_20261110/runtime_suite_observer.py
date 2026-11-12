"""One bounded acceptance owner for a current-runtime suite; never dispatches."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from .contracts import digest, file_digest, require, write_json_atomic
from .runtime_suite import active, checked
from .runtime_suite_acceptance import reconcile, reference
from .storage import atomic_directory_lock


def observe(root, run_dir):
    root=Path(root).resolve();base=Path(run_dir).resolve()
    _,release,manifest,_,current=active(root)
    require(base==current,'acceptance observer is not the active immutable suite')
    state_path=base/'acceptance_observer.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'schema':'isg-runtime-suite-acceptance-observer-v1','classes':{}}
    with atomic_directory_lock(base/'acceptance_observer.lockdir'):
        state.update(run_id=release['run_id'],cycle_started_at=time.time(),owner='/root/dataset_contract')
        for entry in manifest['classes']:
            cls=entry['class_id'];row_path=base/'rows'/(cls+'.json');allocation_path=base/'allocations'/(cls+'.json')
            status={'observed_at':time.time(),'status':'awaiting_authored_row','validation_passed':False}
            try:
                if not row_path.exists():
                    state['classes'][cls]=status;continue
                row_ref=reference(row_path);row=checked(root,row_ref)
                status.update(row=row_ref,status='awaiting_allocation')
                if not allocation_path.exists():
                    state['classes'][cls]=status;continue
                allocation_ref=reference(allocation_path);allocation=checked(root,allocation_ref)
                status.update(allocation=allocation_ref,status='awaiting_native_and_demo')
                if not allocation.get('allocation'):
                    state['classes'][cls]=status;continue
                receipt=checked(root,allocation['dispatch_receipt']);job=allocation['allocation']
                gate_path=(Path(allocation['gate']['path']) if allocation.get('gate') else
                    root/'validation/open_loop'/receipt['allocation']['key']/str(job['jobid'])/'gate.json')
                if not gate_path.exists():
                    status['status']='awaiting_own_replay';state['classes'][cls]=status;continue
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
