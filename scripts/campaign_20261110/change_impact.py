"""Inventory change provenance and executed review evidence without relabeling data.

An inventory, artifact audit, or empty replacement list is never represented as
a completed retrospective physics/scoring review. Eligibility consumes only
explicit, hash-bound decisions; unresolved inventory gaps stay visible.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

from scripts.campaign_20261110.contracts import digest, file_digest, require, write_json_atomic


SCHEMA='isg-change-impact-ledger-v1'
KNOWN={
 '6acd8f1a':('scoring','Round-object yaw symmetry and block-design assignment tie-break; explicit report scoring revision.','rescore'),
 'e4a7e178':('placement_and_execution','Guide-wall placement/endwise goal for under_large02 plus bounded push nudge stall recovery.','investigate'),
 'ed331ccc':('qa','Action replay preserves loaded canonical variant when sidecar is absent for goal-occlusion exemptions.','revalidate'),
 '09a936fe':('acceptance_qa','Require actual native replay QA instead of aggregate gate success alone.','revalidate'),
 'f8707109':('acceptance_qa','Family recovery and claims require actual native replay QA.','revalidate'),
 '23225286':('loading_and_restoration','Restore recorded campaign scene state for action replay and export.','revalidate'),
 'b8aef8cb':('training_eligibility','Standalone training remains excluded until compliant pipeline and distribution proofs.','investigate'),
}


def ref(path):return {'path':str(Path(path).resolve()),'sha256':file_digest(path)}
def now():return datetime.now(timezone.utc).isoformat()


def catalog(repo,baseline,output):
    repo=Path(repo).resolve()
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
    baseline=git('rev-parse',baseline);head=git('rev-parse','HEAD')
    subprocess.run(['git','-C',str(repo),'merge-base','--is-ancestor',baseline,head],check=True)
    changes=[]
    for commit in git('rev-list','--reverse',baseline+'..'+head).splitlines():
        parents=git('show','-s','--format=%P',commit).split();old=parents[0]
        subject=git('show','-s','--format=%s',commit)
        raw=git('diff-tree','--no-commit-id','--no-abbrev','--raw','-r',old,commit)
        files=[]
        for line in raw.splitlines():
            metadata,path=line.split('\t',1);parts=metadata.split()
            files.append({'path':path,'old_blob':parts[2],'new_blob':parts[3],'status':parts[4]})
        paths=[r['path'] for r in files]
        known=next((v for prefix,v in KNOWN.items() if commit.startswith(prefix)),None)
        behavior=any(p.startswith(('ispatialgym/sampling/','ispatialgym/oracle/','ispatialgym/eval/',
            'ispatialgym/robot/','ispatialgym/assets/','ispatialgym/multienv/','scripts/trainfast/')) for p in paths)
        acceptance=any(p.startswith('scripts/campaign_20261110/') and any(t in Path(p).name for t in
            ('acceptance','group_lane','contract','production_gate','eligib','camera_lane','goal_groups','replay_gate')) for p in paths)
        state='impact_scope_pending' if behavior or acceptance or known else 'implementation_review_pending'
        changes.append({'change_id':commit,'old_commit':old,'new_commit':commit,'parents':parents,
            'behavior':known[1] if known else subject,'commit_subject':subject,
            'change_kind':known[0] if known else 'behavior_relevant' if behavior or acceptance else 'candidate_infrastructure_or_observation',
            'action':known[2] if known else 'investigate','review_state':state,'owner':'/root/templates_assets',
            'files':files,'file_diff_sha256':hashlib.sha256(raw.encode()).hexdigest(),
            'scope_established':False,'affected_instance_ids_ref':None,'pending_scope_instance_ids_ref':None,
            'evidence':[],'next_action':'Compare exact old/new behavior with actual instance configuration, switches and data dependencies; classify scope before any review decision.'})
    value={'schema':'isg-change-impact-catalog-v1','campaign_id':'campaign_20261110','observed_at':now(),
        'baseline_commit':baseline,'current_commit':head,'changes':changes,
        'coverage_note':'Every reachable commit after baseline is enumerated, including nonbehavior candidates. Git inventory is not an impact decision.'}
    require(not Path(output).exists(),'immutable catalog destination already exists');write_json_atomic(output,value)
    return value


def native_facts(path,accepted=None,allocation=None,config_cache=None):
    """Extract exact provenance without treating a historical success as a new review."""
    from scripts.campaign_20261110.goal_groups import read_env
    payload=Path(path).read_bytes();doc=json.loads(payload);native=doc.get('ispatialgym',{});block=native.get('campaign',{})
    iid=native.get('instance_id') or (accepted or {}).get('instance_id') or Path(path).name.removesuffix('-tro_state.json')
    row={'instance_id':iid,'native':{'path':str(Path(path).resolve()),'sha256':hashlib.sha256(payload).hexdigest()},
        'class_id':block.get('class_id') or (accepted or {}).get('class_id') or (allocation or {}).get('key'),
        'task':block.get('task') or (accepted or {}).get('task'),'sampling_split':block.get('split'),
        'split':(accepted or {}).get('split') or block.get('split'),'group_id':(accepted or {}).get('group_id') or
            block.get('runtime',{}).get('goal_group',{}).get('group_id'),
        'validation_key':block.get('validation_key') or (accepted or {}).get('validation_key'),
        'template_sha256':block.get('template_sha256'),'perturbations':block.get('perturbations'),
        'runtime_observations':block.get('runtime'),'object_bindings':native.get('object_bindings'),
        'accepted':accepted is not None,'accepted_record_sha256':digest({k:v for k,v in accepted.items() if k!='allocation_source'}) if accepted else None,
        'provenance_complete':bool(block and native.get('instance_id')),'review_status':'not_yet_change_reviewed',
        'original_acceptance_evidence':{k:accepted.get(k) for k in ('report','report_sha256','gate','tro_sha256','sidecar','sidecar_sha256')} if accepted else None}
    job=Path((allocation or {}).get('job_dir') or '') if allocation else next((p for p in Path(path).parents if (p/'job.env').is_file()),None)
    if job and (job/'job.env').is_file():
        env=read_env(job/'job.env');row['job_env']=ref(job/'job.env');row['job_dir']=str(job)
        row['runtime_path']=(allocation or {}).get('code') or env.get('CODE') or env.get('R')
        row['switches']={k:v for k,v in env.items() if (k.startswith('ISPATIALGYM_') or k in
            ('KEEP_SCENE','START_PER_ATTEMPT','ISGF_COLLECT_EXTRA','ISG_CUSTOMS','GRASPS','ISGF_BDDL_ROOT'))
            and not any(word in k.upper() for word in ('TOKEN','PASSWORD','SECRET','CREDENTIAL'))}
        cp=env.get('ISPATIALGYM_CAMPAIGN_CONFIG')
        if cp:
            cache=config_cache if config_cache is not None else {}
            if cp not in cache:
                cfg=json.loads(Path(cp).read_text());cache[cp]={'evidence':ref(cp),'template':cfg.get('template'),
                    'purpose':cfg.get('purpose'),'split':cfg.get('split'),'validation_key':cfg.get('validation_key')}
            row['configuration']=cache[cp]['evidence'];row['configuration_purpose']=cache[cp]['purpose']
            row['template']=cache[cp]['template']
        mp=job/'demo'/iid/'meta.json'
        if mp.exists():
            m=json.loads(mp.read_text());row['original_demo']={'evidence':ref(mp),'success':m.get('success'),
                'reason':m.get('reason'),'qa':m.get('qa'),'error_metric':m.get('error_metric'),
                'scoring_revision':m.get('scoring_revision'),'loader':m.get('loader')}
    metadata=Path(path).parent.parent/'template_meta.json'
    if metadata.exists():
        key=str(metadata);cache=config_cache if config_cache is not None else {}
        if key not in cache:
            m=json.loads(metadata.read_text());facts=m.get('template_facts',{})
            cache[key]={'evidence':ref(metadata),'customs':m.get('customs'),'bddl_sha256':m.get('bddl_sha256'),
                'alias':m.get('alias'),'choice':m.get('choice'),'facts':{k:v for k,v in facts.items() if
                any(s in k for s in ('guide','goal','placement','support','side','variant','scale','articulation','joint'))}}
        row['native_template']=cache[key]
    return row


def inventory(root,catalog_path,output):
    from scripts.campaign_20261110.export_release import snapshot_allocations
    from scripts.campaign_20261110.controller_families import allocations
    root,output=Path(root).resolve(),Path(output).resolve();started=time.time()
    require(output.is_relative_to(root) and not output.exists(),'new campaign review inventory required')
    cat=json.loads(Path(catalog_path).read_text());require(cat['schema']=='isg-change-impact-catalog-v1','unknown catalog')
    accepted,groups,inputs=snapshot_allocations(root);by_id={r['instance_id']:r for r in accepted}
    require(len(by_id)==len(accepted),'duplicate accepted instance ID')
    output.mkdir(parents=True)
    snapshot=output/'accepted_snapshot.json';write_json_atomic(snapshot,{'records':accepted,'groups':groups,'inputs':inputs})
    jobs=allocations(root);job_dirs={str(Path(r['job_dir']).resolve()):r for r in jobs}
    paths={};gaps=[];cache={};seen=set();records=[]
    # Canonical outputs and retained diagnostic/materialized outputs are all
    # indexed. Symlink targets stay inside the campaign; exact copies are joined
    # by stable ID and hash instead of being counted as extra instances.
    roots=[root/'out',root/'materialized_runs',root/'diagnostics']
    for directory in roots:
        if not directory.exists():continue
        for path in directory.rglob('*-tro_state.json'):
            actual=path.resolve()
            if not actual.is_relative_to(root):gaps.append({'path':str(path),'reason':'native path resolves outside campaign'});continue
            if actual.is_file() and actual.stat().st_mtime<=started:paths[str(actual)]=actual
    for row in accepted:
        path=row.get('tro') or row.get('artifacts',{}).get('tro',{}).get('path')
        if path:paths[str(Path(path).resolve())]=Path(path).resolve()
        else:gaps.append({'instance_id':row['instance_id'],'reason':'accepted record has no native artifact'})
    inventory_path=output/'instances.jsonl';line_count=0
    with inventory_path.open('x') as stream:
        for path in sorted(paths.values()):
            try:
                allocation=next((job_dirs[str(p)] for p in path.parents if str(p) in job_dirs),None)
                preliminary=native_facts(path,allocation=allocation,config_cache=cache);iid=preliminary['instance_id']
                accepted_row=by_id.get(iid)
                accepted_native=(accepted_row or {}).get('tro') or (accepted_row or {}).get('artifacts',{}).get('tro',{}).get('path')
                is_accepted_source=bool(accepted_native and Path(accepted_native).resolve()==path)
                if accepted_row and is_accepted_source:
                    preliminary.update(accepted=True,accepted_record_sha256=digest({k:v for k,v in accepted_row.items() if k!='allocation_source'}),
                        split=accepted_row['split'],group_id=accepted_row.get('group_id'),
                        original_acceptance_evidence={k:accepted_row.get(k) for k in ('report','report_sha256','gate','tro_sha256','sidecar','sidecar_sha256')})
                    expected=accepted_row.get('tro_sha256') or accepted_row.get('artifacts',{}).get('tro',{}).get('sha256')
                    preliminary['accepted_source_integrity']=expected==preliminary['native']['sha256']
                    if not preliminary['accepted_source_integrity']:gaps.append({'instance_id':iid,'path':str(path),'reason':'accepted native hash differs'})
                    seen.add(iid)
                elif accepted_row:
                    preliminary['same_id_accepted_source']=str(accepted_native)
                    preliminary['acceptance_relation']='retained copy or diagnostic; not the canonical accepted source'
                preliminary['allocation']={k:allocation.get(k) for k in ('jobid','key','code','start','end','ledger')} if allocation else None
                stream.write(json.dumps(preliminary,separators=(',',':'))+'\n');line_count+=1
                records.append({'instance_id':iid,'native_sha256':preliminary['native']['sha256'],'accepted':is_accepted_source,
                    'accepted_record_sha256':preliminary['accepted_record_sha256'],'class_id':preliminary['class_id'],
                    'split':preliminary['split'],'group_id':preliminary['group_id'],'provenance_complete':preliminary['provenance_complete']})
            except (OSError,ValueError,KeyError,TypeError) as exc:gaps.append({'path':str(path),'reason':type(exc).__name__+': '+str(exc)})
            if line_count and line_count%250==0:
                stream.flush();write_json_atomic(output/'progress.json',{'observed_at':now(),'indexed_native_files':line_count,
                    'native_files_discovered':len(paths),'accepted_seen':len(seen),'gaps':len(gaps),'status':'inventory_running'})
    for iid in sorted(set(by_id)-seen):gaps.append({'instance_id':iid,'reason':'accepted instance missing from readable native inventory'})
    identities=defaultdict(set)
    for row in records:identities[row['instance_id']].add(row['native_sha256'])
    conflicts=[{'instance_id':iid,'native_sha256':sorted(hashes)} for iid,hashes in identities.items() if len(hashes)>1]
    counts=Counter(r['instance_id'] for r in records)
    manifest={'schema':'isg-change-impact-instance-inventory-v1','campaign_id':'campaign_20261110','observed_at':now(),
        'capture_started_unix':started,'catalog':ref(catalog_path),'accepted_snapshot':ref(snapshot),'native_roots':[str(p) for p in roots],
        'instances':ref(inventory_path),'index':records,'accepted_records':len(accepted),'accepted_indexed':len(seen),
        'generated_unique_ids':len(identities),'native_artifact_records':line_count,
        'generated_unaccepted':len(set(identities)-set(by_id)),'duplicate_id_artifacts':sum(n-1 for n in counts.values()),
        'conflicting_native_ids':conflicts,'gaps':gaps,'completeness':'explicit roots and accepted snapshot enumerated; allocation/native-count reconciliation pending',
        'review_complete':False,'historical_pass_is_change_review':False}
    write_json_atomic(output/'inventory.json',manifest)
    ledger={'schema':SCHEMA,'campaign_id':'campaign_20261110','observed_at':now(),'baseline_commit':cat['baseline_commit'],
        'current_commit':cat['current_commit'],'inventory':{'manifest':ref(output/'inventory.json'),
            'accepted_snapshot_sha256':file_digest(snapshot),'completeness':manifest['completeness'],'gaps':gaps},
        'summary':{'accepted_indexed':len(seen),'generated_indexed':len(identities),'generated_unaccepted':manifest['generated_unaccepted'],
            'changes_total':len(cat['changes']),'changes_pending':len(cat['changes']),'verified':0,'under_review':0,'excluded':0,
            'inventory_gaps':len(gaps),'review_coverage_complete':False},
        'changes':cat['changes'],'decisions':[],'executed_reviews':[],
        'coverage':{'evaluated_change_ids':[],'pending_change_ids':[c['change_id'] for c in cat['changes']],
                    'explicitly_evaluated_instance_ids_ref':None,'complete':False},
        'notice':'Inventory is preparation. Empty decisions/replacements do not mean reviewed or compatible. Historical acceptance remains preserved.'}
    write_json_atomic(output/'ledger_initial.json',ledger)
    current=root/'validation/change_impact/ledger.json'
    if not current.exists():write_json_atomic(current,ledger)
    else:write_json_atomic(output/'publication_pending.json',{'reason':'existing live ledger retained; merge new inventory with prior decisions before publication'})
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    c=sub.add_parser('catalog');c.add_argument('--repo',type=Path,required=True);c.add_argument('--baseline',default='9949b012');c.add_argument('--output',type=Path,required=True)
    i=sub.add_parser('inventory');i.add_argument('--root',type=Path,required=True);i.add_argument('--catalog',type=Path,required=True);i.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.mode=='catalog':r=catalog(a.repo,a.baseline,a.output);print(json.dumps({'changes':len(r['changes']),'output':str(a.output)}))
    else:r=inventory(a.root,a.catalog,a.output);print(json.dumps({k:r[k] for k in ('accepted_records','accepted_indexed','generated_unique_ids','generated_unaccepted','native_artifact_records','completeness')}|{'gaps':len(r['gaps'])}))


if __name__=='__main__':main()
