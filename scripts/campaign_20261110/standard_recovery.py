"""Read-only standard coverage ledger; generation never dispatches collection work."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from scripts.campaign_20261110.contracts import PENDING_ALIASES, file_digest, write_json_atomic
from scripts.campaign_20261110.controller_families import allocations
from scripts.campaign_20261110.production_gate import PARKED_CLASSES
from scripts.campaign_20261110.goal_groups import read_env
from scripts.campaign_20261110.replay_gate_quality import assess

CAMPAIGN='campaign_20261110'
DATASET={'configuration_matching-articulation_open_small_scale_01','configuration_matching-attachment_02',
 'inter_object_relation_satisfaction-inside_01','mental_rotation-soup_ladle','mental_rotation-toy_figure',
 'object_scaling-beefsteak_tomato','object_scaling-bottle_of_pop',*(f'disembedding_{i:02}' for i in range(7,11))}
ROOT={*(f'disembedding_{i:02}' for i in [1,3,4,5,6]),
 *(f'inter_object_relation_satisfaction-inside_{i:02}' for i in [2,3,4]),
 *(f'inter_object_relation_satisfaction-nextto_{i:02}' for i in [1,2]),
 *(f'inter_object_relation_satisfaction-touching_{i:02}' for i in [1,2,3,4])}
COMPUTE={*(f'inter_object_relation_satisfaction-inside_large_scale_{i:02}' for i in [1,2,3]),
 *(f'inter_object_relation_satisfaction-ontop_large_scale_{i:02}' for i in [1,2,3]),
 'inter_object_relation_satisfaction-touching_large_scale_01',
 'inter_object_relation_satisfaction-under_large_scale_02'}

def owner(key):
 if key in PARKED_CLASSES:return 'user_artifacts; tracking /root'
 if key in {'disembedding_01','disembedding_05','inter_object_relation_satisfaction-touching_02'}:return '/root/dataset_contract'
 if key=='inter_object_relation_satisfaction-nextto_01':return '/root/templates_assets'
 if key in DATASET:return '/root/dataset_contract'
 if key in ROOT:return '/root'
 if key in COMPUTE:return '/root/compute'
 return '/root/templates_assets'

def read(path,default=None):
 try:return json.loads(Path(path).read_text())
 except (OSError,ValueError):return default

def identity(path):
 path=Path(path)
 try:return {'path':str(path),'sha256':file_digest(path),'mtime':path.stat().st_mtime}
 except OSError:return {'path':str(path),'unavailable':True}

def received_artifacts(root):
 """Only explicit owner updates backed by exact readiness audits lift a park."""
 raw=read(root/'infra/standard_recovery/owner_artifact_updates.json',{}) or {}
 if not raw:return {}
 if raw.get('campaign_id')!=CAMPAIGN:raise ValueError('artifact update belongs to another campaign')
 result={}
 gate=read(root/'controllers/training_production_gate.json',{}) or {}
 current_holds=set(gate.get('parked_classes',[])) if gate.get('campaign_id')==CAMPAIGN else set()
 for row in raw.get('artifacts',[]):
  key=row['class_id'];proof=Path(row['audit']['path']).resolve()
  if key in current_holds:continue  # A newer explicit owner hold supersedes an earlier availability proof.
  if key not in PENDING_ALIASES:raise ValueError('artifact update is not a parked class')
  if not proof.is_relative_to(root/'infra/standard_recovery'):raise ValueError('artifact audit outside campaign recovery')
  if file_digest(proof)!=row['audit']['sha256']:raise ValueError('artifact readiness audit changed')
  audit=read(proof)
  matches=[r for r in audit.get('classes',[]) if r['class_id']==key]
  if audit.get('campaign_id')!=CAMPAIGN or len(matches)!=1:raise ValueError('artifact audit class differs')
  record=matches[0]
  if not (record.get('all_ready_and_exact_assets') is True and record.get('exact_source_template') is True
          and record.get('errors')==[] and record.get('reservations')=='passed'):
   raise ValueError('artifact source/readiness/reservation audit incomplete')
  if not row.get('owner_instruction') or not row.get('source_commit'):raise ValueError('artifact update lacks owner provenance')
  result[key]=row
 return result

def configured_attempt(root,row,validation,cluster):
 job=Path(row['job_dir']); cfg=None; cp=None
 if not job.resolve().is_relative_to(root/'out'):return None
 for ep in [job/'job.env.before_vast_caches',job/'job.env']:
  if not ep.exists():continue
  env=read_env(ep);name=env.get('ISPATIALGYM_CAMPAIGN_CONFIG')
  if not name:continue
  p=Path(name)
  if p.resolve().is_relative_to(root):
   candidate=read(p)
   if candidate is not None:cfg=candidate;cp=p;break
 if cfg is None or cfg.get('split')!='eval_standard' or cfg.get('template',{}).get('class_id')!=row['key']:return None
 stats=read(job/'job_stats.json',{}) or {}
 jid=row['jobid']; vr=validation.get('gates',{}).get(jid,{})
 gp=root/'validation/open_loop'/row['key']/jid/'gate.json';gate=read(gp)
 replay={'status':vr.get('status','unavailable'),'gate':identity(gp)}
 if gate is not None:replay.update(quality=assess(gate,gp),summary_success=gate.get('success'),mode=gate.get('replay_mode'))
 state=read(root/'vast/state/jobs'/f'{jid}.json',{}) or {}
 active=state.get('status') or cluster.get(jid,{}).get('state')
 runtime=Path(row['code']); commitment=runtime/'.snapshot_commit';actual=identity(commitment)
 return {'job_id':jid,'job_dir':str(job),'index':int(row['start']),'requested':int(row['n']),
  'configuration':{**identity(cp),'purpose':cfg.get('purpose'),'validation_key':cfg.get('validation_key'),
    'template_id':cfg['template'].get('template_id'),'class_id':row['key']},
  'runtime':{'snapshot':str(runtime),'commit':commitment.read_text().strip() if commitment.exists() else None,
    'identity':actual,'matches_configuration':actual.get('sha256')==cfg.get('validation_key',{}).get('code_sha256')},
  'native':{'attempts':stats.get('attempts'),'written':stats.get('written'),'rejections':stats.get('rejections'),
    'status':'passed' if stats.get('written',0)>0 else 'failed' if stats.get('attempts',0)>0 else 'unavailable_or_setup_only',
    'evidence':identity(job/'job_stats.json')},
  'demo':{'reported_attempts':stats.get('demos'),'reported_successes':stats.get('successes'),
    'status':'reported_pass' if stats.get('successes',0)>0 else 'failed' if stats.get('demos',0)>0 else 'not_produced',
    'note':'Demo summary alone does not establish hard QA, runtime compatibility, replay, or sibling acceptance.'},
  'replay':replay,'job_state':active,'validation_status':vr.get('status'),
  'last_progress_at':max([x for x in [vr.get('updated_at'),gp.stat().st_mtime if gp.exists() else None,
    (job/'job_stats.json').stat().st_mtime if (job/'job_stats.json').exists() else None] if isinstance(x,(int,float))],default=None)}

def state_from_attempts(attempts):
 if any(a.get('job_state') in ('RUNNING','assigned','staging','running') for a in attempts):return 'running'
 if any(a.get('job_state') in ('PENDING','queued') for a in attempts):return 'queued'
 if attempts:return 'failed'
 return 'unattempted'

def combine(baseline,evidence,technical,workers,previous=None,artifact_updates=None,eligibility=None):
 """Acceptance is authoritative; a worker cannot promote a demo to acceptance."""
 now=datetime.now(timezone.utc).isoformat();previous=previous or {};artifact_updates=artifact_updates or {}
 current={r['class_id']:r for r in evidence['classes'] if r['split']=='eval_standard'}
 historical={k:dict(v) for k,v in current.items()}
 if eligibility is not None:
  if eligibility.get('campaign_id')!=CAMPAIGN or eligibility.get('schema')!='isg-current-eligibility-v1':raise ValueError('invalid current eligibility projection')
  eligible={r['class_id']:r for r in eligibility['classes'] if r['split']=='eval_standard'}
  current={k:{**r,'accepted':eligible.get(k,{}).get('current_eligible',0)} for k,r in current.items()}
 initial={r['class_id']:r for r in baseline['classes']}
 baseline_size=len(initial)
 for key,c in current.items():
  if c['accepted']==0 and key not in initial:initial[key]={**c,'accepted':None}
 acceptance_time=eligibility['observed_at'] if eligibility is not None else evidence.get('observed_at')
 prior={r['class_id']:r for r in previous.get('classes',[])}
 notes={}
 for doc in workers:
  if doc.get('campaign_id')!=CAMPAIGN:continue
  rs=doc.get('classes',[]); rs=list(rs.values()) if isinstance(rs,dict) else rs
  for row in rs:notes[row['class_id']]={**row,'worker_observed_at':doc.get('observed_at'),'responsible_worker':doc.get('owner')}
 rows=[]
 for key,b in sorted(initial.items()):
  c=current.get(key);count=c.get('accepted') if c else None
  attempts=technical.get(key,[]);w=notes.get(key,{})
  parked=key in PARKED_CLASSES
  state=w.get('state',state_from_attempts(attempts))
  if parked:state='parked'
  elif count and count>0:state='accepted'
  elif state in ('accepted','parked'):state='failed'
  first=prior.get(key,{}).get('first_accepted_observed_at')
  if count and count>0 and not first:first=acceptance_time
  latest=attempts[0] if attempts else {}
  rows.append({'class_id':key,'task':b['task'],'accepted_standard':count,'target':10,
    'remaining':max(0,10-count) if count is not None else None,'baseline_accepted':b['accepted'],
    'historical_accepted_standard':historical.get(key,{}).get('accepted'),
    'first_accepted_observed_at':first,'state':state,'owner':artifact_updates.get(key,{}).get('owner',owner(key)),
    'artifact_update':artifact_updates.get(key),
    'canonical_validation_eligible':False if parked else None,
    'engineering_status':'awaiting_promised_user_artifact' if parked else w.get('engineering_status','assigned_for_diagnosis'),
    'blocker':'Promised updated oracle/template artifact has not arrived.' if parked else w.get('blocker',
      'Source artifact and readiness audit received; fresh native/demo/replay and sibling proof still required.' if key in artifact_updates else (c or {}).get('reason','Evidence unavailable')),
    'next_action':'Resume only after the promised artifact arrives; continue independent classes.' if parked else w.get('next_action','Read exact native/demo failure and implement a bounded correction; do not repeat unchanged failed configurations.'),
    'configuration':w.get('configuration') or latest.get('configuration'), 'runtime':w.get('runtime') or latest.get('runtime'),
    'native':w.get('native') or latest.get('native'), 'demo':w.get('demo') or latest.get('demo'), 'replay':w.get('replay') or latest.get('replay'),
    'acceptance':{'accepted_standard':count,'observed_at':acceptance_time,'source':'current eligibility projection over locked allocation audit' if eligibility is not None else 'locked allocation audit',
      'change_impact':next((r for r in eligibility['classes'] if r['class_id']==key and r['split']=='eval_standard'),{}) if eligibility is not None else None,
      'worker_analysis':w.get('acceptance'),'required':'Exact physics replay + hard QA + compatible configuration + a distinct accepted training sibling'},
    'active_jobs':w.get('active_jobs',[]),'controller':w.get('controller',[]),
    'diagnosis':w.get('diagnosis',{'status':'reference_comparison_pending',
      'next_discriminating_experiment':'Verify the indexed successful configuration and first failing campaign stage before another change.'}),
    'last_progress_at':w.get('last_progress_at') or latest.get('last_progress_at'),
    'worker_observed_at':w.get('worker_observed_at'),'technical_attempts':attempts,
    'evidence':w.get('evidence',(c or {}).get('reason_evidence',[]))})
 tasks=Counter()
 for c in current.values():tasks[c['task']]+=c['accepted']
 return {'schema':'isg-standard-recovery-v1','campaign_id':CAMPAIGN,'observed_at':now,
  'acceptance_observed_at':acceptance_time,'baseline_observed_at':baseline['observed_at'],
  'current_eligibility':{'ledger':eligibility.get('ledger'),'review_inventory_complete':eligibility.get('review_inventory_complete')} if eligibility is not None else None,
  'baseline_zero_classes':baseline_size,'zero_classes':sum(c['accepted']==0 for c in current.values()),
  'covered_classes':sum(c['accepted']>0 for c in current.values()),'standard_classes':len(current),
  'zero_tasks':[t for t,n in tasks.items() if n==0],
  'newly_covered':[r['class_id'] for r in rows if r['accepted_standard'] and r['baseline_accepted']==0],
  'state_counts':dict(Counter(r['state'] for r in rows)),'classes':rows,
  'activity_semantics':'Worker observation timestamps identify active engineering reports. Persistent automation and queued jobs do not establish ongoing agent inspection or accepted coverage.',
  'task_semantics':'The original benchmark defines7concept families and23tasks. Zero task counts refer to tasks, not concept families.'}

def report(data):
 from zoneinfo import ZoneInfo
 stamp=datetime.fromisoformat(data['observed_at']).astimezone(ZoneInfo('America/Chicago')).isoformat()
 lines=['STANDARD EVAL RECOVERY — campaign_20261110','Observed: '+stamp,
  f"Accepted coverage: {data['covered_classes']}/{data['standard_classes']} classes; {data['zero_classes']} remain zero.",
  'First accepted since recovery baseline: '+(', '.join(data['newly_covered']) or 'none yet'),
  'Zero tasks: '+', '.join(data['zero_tasks']),data['activity_semantics'],'']
 for r in data['classes']:
  lines += [f"{r['class_id']} — {r['state']} — accepted {r['accepted_standard']}/10",
   f"Owner: {r['owner']}; engineering: {r['engineering_status']}; worker observation: {r['worker_observed_at']}; last progress: {r['last_progress_at']}",
   'Blocker: '+str(r['blocker']),'Next action: '+str(r['next_action']),
   'Active jobs/controller: '+json.dumps([r['active_jobs'],r['controller']]),
   'Exact configuration/runtime: '+json.dumps([r['configuration'],r['runtime']]),
   'Native/demo/replay/acceptance: '+json.dumps([r['native'],r['demo'],r['replay'],r['acceptance']]),'']
  lines += ['Reference comparison / causal diagnosis: '+json.dumps(r.get('diagnosis',{})), '']
 return '\n'.join(lines)+'\n'

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--once',action='store_true');p.add_argument('--interval',type=int,default=120);a=p.parse_args()
 root=a.root.resolve();directory=root/'infra/standard_recovery';destination=directory/'ledger.json'
 while True:
  try:
   baseline=read(directory/'baseline.json');evidence=read(root/'infra/dashboard_evidence.json')
   assert baseline['campaign_id']==evidence['campaign_id']==CAMPAIGN
   targets={r['class_id'] for r in baseline['classes']};technical=defaultdict(list)
   validation=read(root/'validation/controller.json',{});cluster=read(root/'infra/cluster_snapshot.json',{})
   live={str(j['id']):j for j in cluster.get('jobs',[])}
   unique={r['jobid']:r for r in allocations(root) if r['key'] in targets}
   for row in sorted(unique.values(),key=lambda r:int(r['start']),reverse=True):
    if len(technical[row['key']])>=3:continue
    try:
     rec=configured_attempt(root,row,validation,live)
     if rec is not None:technical[row['key']].append(rec)
    except (OSError,ValueError,KeyError,TypeError):continue
   workers=[read(f,{}) for f in sorted(directory.glob('worker_*.json'))]
   value=combine(baseline,evidence,technical,workers,read(destination,{}),received_artifacts(root),read(root/'accounting/current_eligibility.json'))
   write_json_atomic(destination,value)
   target=directory/'REPORT.txt';temp=target.with_suffix('.tmp');temp.write_text(report(value));temp.replace(target)
   print(json.dumps({k:value[k] for k in ['observed_at','covered_classes','zero_classes','newly_covered','state_counts']}),flush=True)
  except Exception as error:
   write_json_atomic(directory/'error.json',{'observed_at':datetime.now(timezone.utc).isoformat(),'error':type(error).__name__+': '+str(error)})
   if a.once:raise
  if a.once:return
  time.sleep(max(30,a.interval))

if __name__=='__main__':main()
