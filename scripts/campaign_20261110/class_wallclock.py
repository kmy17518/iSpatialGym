"""Measured class budgets; no calibration GPU jobs and no mid-group changes."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
import time

SCHEMA='isg-class-wallclock-v1'


def quantile(values,q):
    values=sorted(float(x)for x in values if math.isfinite(float(x))and float(x)>0)
    return values[min(len(values)-1,math.ceil(q*len(values))-1)]if values else None


def build(csv_path, online_states=()):
    observed={}
    for path in online_states:
        state=json.loads(Path(path).read_text())
        for group in state['groups']:
            if not group.get('goal'):continue
            cid=group['goal']['class_id'];start=group['started_at']
            successes=sorted(m['evidence']['result']['finished_at_unix']-start for m in group['members']if m['status']=='accepted')
            row={'source':str(path),'group_id':group['id'],'group_seconds':group['finished_at']-start,
                 'second_success_s':successes[1]if len(successes)>1 else None,'qualifies':group['qualifies'],
                 'deadline_s':group['deadline']-start,'member_seconds':[m['evidence']['result']['wall_s']for m in group['members']if m['status']=='accepted']}
            observed.setdefault(cid,[]).append(row)
    classes={}
    for row in csv.DictReader(Path(csv_path).open()):
        cid=row['class_id'];samples=int(row['source_timing_samples']);historical=float(row['planning_minutes_per_group'])*60
        groups=observed.get(cid,[]);second=[g['second_success_s']for g in groups if g['second_success_s']is not None]
        members=[t for g in groups for t in g['member_seconds']]
        # Initial source-job timings include startup and collection, hence are
        # conservative. Fresh-goal generation is measured during production.
        eval_s=math.ceil(max(900,(quantile([g['group_seconds']for g in groups if g['qualifies']],.9)or historical)*1.25)/60)*60
        # For training, the median observed time to a second accepted member is
        # the initial 50% target; the observed censored fraction is explicit.
        train_s=math.ceil((quantile(second,.5)or historical*.5)/60)*60
        member_s=math.ceil(max(300,(quantile(members,.95)*2 if members else historical*.5))/60)*60
        classes[cid]={'eval_standard':{'group_limit_s':eval_s,'member_limit_s':min(eval_s,member_s),'minimum_members':2,'target_members':3},
          'train':{'group_limit_s':min(eval_s,max(300,train_s)),'member_limit_s':min(eval_s,max(300,train_s),member_s),'minimum_members':1,'target_members':3,'multi_member_yield_target':.5},
          'eval_model':{'group_limit_s':min(eval_s,max(600,member_s)),'member_limit_s':min(eval_s,max(600,member_s)),'minimum_members':1,'target_members':1},
          'evidence':{'historical_source_samples':samples,'historical_observed_p90_source_minutes':row['observed_p90_source_minutes']or None,
                      'historical_source_job_ids':row['registered_source_jobs'].split(','),'online_groups':groups,
                      'online_censored_groups':sum(not g['qualifies']for g in groups),'source_csv_sha256':hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()},
          'uncertainty':{'fresh_goal_runtime_measured':bool(groups),'training_50_percent_measured':False,
                         'member_limit_basis':'online oracle p95 x2 with sampling allowance'if members else 'half historical group-source allowance; update from generation',
                         'fallback':None if samples or groups else row['planning_basis']}}
    return {'schema':SCHEMA,'created_at':time.time(),'classes':classes,'semantics':{'eval':'target three; at least two distinct passing members',
            'train':'shorter initial per-class budgets; measure actual >1-member fraction, including failures','heldout':'singletons eligible; target one',
            'updates':'new immutable policy per subsequent worker; active group deadlines never reset','gpu_speed':'record actual GPU/driver and timing; no unmeasured speed multiplier'}}


def resolve(reference,cid,split):
    path=Path(reference['path']);raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=reference['sha256']:raise ValueError('class timeout policy changed')
    policy=json.loads(raw)
    if policy['schema']!=SCHEMA or cid not in policy['classes']:raise ValueError('missing class timeout policy')
    mode='eval_model'if split=='eval_category'else split
    result=policy['classes'][cid][mode]
    if result['group_limit_s']<=0 or not 0<result['member_limit_s']<=result['group_limit_s']:raise ValueError('invalid class deadlines')
    return result
