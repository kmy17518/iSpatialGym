"""Strictly read terminal floating validators without changing their artifacts."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from scripts.campaign_20261110.contracts import file_digest, require, write_json_atomic
try:
    from scripts.campaign_20261110.audit_selected_libraries import validate_record, reference
except ModuleNotFoundError:
    # Standalone reader bundles can audit older native snapshots without editing them.
    from audit_selected_libraries import validate_record, reference
from scripts.campaign_20261110.reservation_policy import active_policy


def validate_arm(prepared, library, validation, candidates):
    expected=prepared['row']['validation_override']['arm']
    require(expected in ('left','right'), 'missing validating arm')
    require(library.get('arm')==validation.get('config',{}).get('arm')==candidates.get('arm')==expected,
            'validator/library/candidate arm mismatch')
    require(library.get('policy',{}).get('arm')==expected, 'contact policy arm mismatch')
    require(library.get('policy_key')==validation.get('policy_key'), 'validator contact policy mismatch')
    require(validation.get('validator')=='isg-floating-validator/2','not floating physics validation')
    tasks=prepared['row']['job']['held_tasks']
    require(set(library.get('held_tasks',[]))==set(tasks)==set(validation.get('tasks',[])), 'held task requirements differ')
    valid={(t.get('candidate'),t.get('task')) for t in validation.get('trials',[])
           if t.get('verdict')=='valid' and t.get('motions_executed',0)>0
           and any(m.get('executed') and m.get('held') and m.get('outcome')=='ok' for m in t.get('motions',[]))}
    require(all((g['name'],task)in valid for g in library['grasps'] for task in tasks),
            'library grasp lacks a successful exact validator motion trial')
    return expected


def audit(root, reconciliation, policy_root=None):
    from ispatialgym.grasping import schema as S
    from ispatialgym.sampling.knowledge import is_ready
    pointer,policy,_=active_policy(policy_root or root)
    rows=[]
    for original in reconciliation['rows']:
        row={**original}
        try:
            for kind in ('prepared','terminal','library','validation'):
                require(file_digest(row[kind]['path'])==row[kind]['sha256'],kind+' evidence changed')
            prepared=json.loads(Path(row['prepared']['path']).read_text());job=prepared['row']['job']
            library_path=Path(row['library']['path']);library=json.loads(library_path.read_text())
            validation=json.loads(Path(row['validation']['path']).read_text())
            cand=prepared['candidates_after'];require(file_digest(cand['path'])==cand['sha256'],'candidate evidence changed')
            candidates=json.loads(Path(cand['path']).read_text())
            for part in prepared['generator_components']:
                require(file_digest(part['path'])==part['sha256'],'generator component changed')
            require(file_digest(job['asset_path'])==job['asset_sha256'] and is_ready(job['model']), 'asset/readiness changed')
            arm=validate_arm(prepared,library,validation,candidates)
            row.update(arm=arm,job=job,selected_dependency_ids=prepared['row']['selected_dependency_ids'],
                       consumers=prepared['row']['consumers'],generator_components=prepared['generator_components'])
            if not library['grasps']:
                row.update(status='failed',reason='native validator returned zero positive grasps',native_reader_passed=False)
            else:
                count=validate_record(job,library,policy)
                library_root=library_path.parents[2]
                native=S.load_model(job['category'],job['model'],job['scale'],str(library_root),constraint='floating')
                require(native is not None and Path(native.path).resolve()==library_path.resolve(),'native resolver route mismatch')
                S.check_compatible(native,scale=job['scale'])
                require(len(native.grasps())==count,'native positive count differs')
                row.update(status='verified',native_reader_passed=True,native_identity=native.identity(),
                           native_identity_key=native.identity_key(),positive_grasps=count,
                           physics_profile=library['physics_profile'],policy_key=library['policy_key'])
        except (OSError,ValueError,KeyError,TypeError) as exc:
            row.update(status='unavailable' if isinstance(exc,OSError) else 'failed',reason=str(exc),native_reader_passed=False)
        rows.append(row)
    return {'schema':'isg-floating-arm-native-reader-v1','observed_at':datetime.now(timezone.utc).isoformat(),
            'owner':'/root/templates_assets','native_reader':reference(S.__file__),'reservation_policy':pointer,
            'rows':rows,'summary':dict(Counter(r['status']for r in rows)),'production_authorized':False,'benchmark_credit':0}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','reconciliation','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--policy-root',type=Path)
    a=p.parse_args();require(not a.output.exists()and a.output.resolve().is_relative_to(a.root.resolve()),'fresh owned output required')
    result=audit(a.root,json.loads(a.reconciliation.read_text()),a.policy_root);result['source']=reference(a.reconciliation)
    write_json_atomic(a.output,result);print(json.dumps({'output':reference(a.output),'summary':result['summary']}))

if __name__=='__main__':main()
