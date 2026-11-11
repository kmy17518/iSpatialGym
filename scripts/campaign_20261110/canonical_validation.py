"""Audit one full accepted standard group per canonical class; no dispatch actions."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from scripts.campaign_20261110.contracts import PENDING_ALIASES, digest, require, write_json_atomic
from scripts.campaign_20261110.dashboard_evidence import standard_catalog
from scripts.campaign_20261110.export_release import ArtifactAudit, normalize_record, snapshot_allocations, template_from
from scripts.campaign_20261110.reservation_policy import active_policy
from scripts.campaign_20261110.standard_recovery import owner

CAMPAIGN='campaign_20261110'


def audit(root, catalog, auditor, started_at):
    records, groups, sources=snapshot_allocations(root)
    policy=active_policy(root)
    require(policy is not None,'active reservation policy unavailable')
    by_class=defaultdict(list);by_group=defaultdict(list)
    for row in records:
        if row.get('group_id'):by_group[row['group_id']].append(row)
        if row['split']=='eval_standard':
            cls=row.get('class_id') or row.get('contract',{}).get('class_id')
            by_class[cls].append(row)
    rows=[]
    templates={}
    for cls,task in sorted(catalog.items()):
        row=dict(class_id=cls,task=task,validated=False,owner=owner(cls),
                 historical_accepted_standard=len(by_class[cls]),status='awaiting_complete_proof',
                 evidence=[],errors=[],next_action='Diagnose first failing stage and validate an exact configuration with its own replay and distinct sibling.')
        if cls in PENDING_ALIASES:
            row.update(status='parked_owner_artifact',next_action='Await owner completion confirmation and replacement oracle/template; retain historical outcomes separately.')
            rows.append(row);continue
        for accepted in reversed(by_class[cls]):
            try:
                members=by_group[accepted['group_id']] if accepted.get('group_id') else [accepted]
                camera=task.startswith('camera_relocalization-')
                if not camera:
                    group=groups[accepted['group_id']]
                    require(2<=len(members)<=3,'standard goal group lacks2–3 successful accepted members')
                    require(group['accepted']=={r['instance_id']:r['split'] for r in members},'accepted group is incomplete')
                    require(sum(r['split']=='eval_standard' for r in members)==1 and any(r['split']=='train' for r in members),'required accepted training sibling missing')
                    require(len({r['instance_id'] for r in members})==len(members),'duplicate accepted member IDs')
                checked=[]
                for member in members:
                    if not member.get('contract'):
                        cfg=json.loads((root/'groups'/member['group_id']/'campaign_config.json').read_text())
                        template=template_from(cfg['template']);templates[template.fingerprint]=template
                    normalized,template=normalize_record(member,root=root,started_at=started_at,templates=templates,
                        reservations=policy[1],reservation_hashes=policy[2],auditor=auditor)
                    require(template.class_id==cls and template.split=='eval_standard','member belongs to another class or sampling split')
                    checked.append(normalized)
                require(len({json.dumps(r['validation_key'],sort_keys=True) for r in checked})==1,'group validation identity differs')
                require(len({json.dumps(r['perturbations'],sort_keys=True) for r in checked})==1,'group perturbations differ')
                if not camera:
                    native_groups=[json.loads(Path(r['artifacts']['tro']['path']).read_text())['ispatialgym']['campaign']['runtime']['goal_group'] for r in checked]
                    require(all(g.get('requested_goal') and g.get('requested_start') for g in native_groups),'native intended goal/start evidence missing')
                    require(len({digest(g['requested_goal']) for g in native_groups})==1,'native requested goals differ')
                    require(len({digest(g['requested_start']) for g in native_groups})==len(native_groups),'native starts/grasps are not distinct')
                row.update(validated=True,status='fresh_chain_verified',group_id=accepted.get('group_id'),
                    camera_group_exempt=camera,validation_key=accepted['validation_key'],
                    configuration=checked[0]['template'],
                    evidence=[dict(instance_id=r['instance_id'],split=r['split'],gate=r['gate'],
                        artifacts=r['artifacts'],validation=r['evidence'])for r in checked],
                    next_action='Generate only remaining strict standard-eval quota using this exact validated configuration.')
                break
            except (OSError,ValueError,KeyError,TypeError) as exc:
                row['errors'].append(dict(instance_id=accepted['instance_id'],reason=type(exc).__name__+': '+str(exc)))
        rows.append(row)
    require(len(rows)==100,'canonical validation denominator must be100')
    return dict(campaign_id=CAMPAIGN,observed_at=datetime.now(timezone.utc).isoformat(),
        schema='isg-canonical-template-validation-v1',total=100,validated=sum(r['validated']for r in rows),
        state_counts=dict(Counter(r['status']for r in rows)),classes=rows,sources=sources,
        semantics='One exact configuration per canonical class verified from original accepted artifacts, demo and native replay evidence; noncamera classes include accepted training siblings. Historical accepted counts are separate. Five current owner-artifact holds are excluded.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--catalog',type=Path,required=True);p.add_argument('--started-at',type=float,required=True)
    p.add_argument('--once',action='store_true');p.add_argument('--interval',type=int,default=180)
    a=p.parse_args();root=a.root.resolve();catalog=standard_catalog(a.catalog)
    auditor=ArtifactAudit(root,a.started_at);out=root/'infra/standard_recovery/canonical_validation.json'
    while True:
        try:
            value=audit(root,catalog,auditor,a.started_at);write_json_atomic(out,value)
            print(json.dumps({k:value[k]for k in ('observed_at','validated','total','state_counts')}),flush=True)
        except Exception as exc:
            write_json_atomic(out.with_name('canonical_validation_error.json'),dict(campaign_id=CAMPAIGN,
                observed_at=datetime.now(timezone.utc).isoformat(),error=type(exc).__name__+': '+str(exc)))
            if a.once:raise
        if a.once:break
        time.sleep(max(30,a.interval))


if __name__=='__main__':main()
