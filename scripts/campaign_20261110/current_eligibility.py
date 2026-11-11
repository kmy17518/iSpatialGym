"""Apply explicit change-impact decisions without rewriting acceptance history.

An unfinished inventory is visible separately; it never becomes a fabricated
completed review or an indiscriminate exclusion of unrelated instances.
"""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import time
from scripts.campaign_20261110.contracts import digest,file_digest,require,write_json_atomic


_LEDGERS = {}
_INDEXES = {}


def read_ledger(root):
    path=Path(root)/'validation/change_impact/ledger.json'
    if not path.exists():return None,None
    stat=path.stat();signature=(stat.st_ino,stat.st_size,stat.st_mtime_ns)
    previous=_LEDGERS.get(str(path))
    if previous and previous[0]==signature:return previous[1],previous[2]
    value=json.loads(path.read_text())
    require(value.get('schema')=='isg-change-impact-ledger-v1' and value.get('campaign_id')=='campaign_20261110','invalid change-impact ledger')
    ref={'path':str(path),'sha256':file_digest(path)}
    _LEDGERS[str(path)]=(signature,value,ref)
    return value,ref


def clean(row):return {k:v for k,v in row.items() if k!='allocation_source'}


def decision_status(root,row,decisions):
    matches=decisions.get(row['instance_id'],[])
    if not matches:return {'status':'scope_not_reconciled','quota_eligible':True,'reasons':['no completed explicit change-impact decision']}
    outcomes=[]
    for d in matches:
        status=d.get('status');reasons=[]
        if d.get('accepted_record_sha256')!=digest(clean(row)):
            status='under_review';reasons.append('accepted record identity is not reconciled with impact decision')
        if status=='verified':
            try:
                refs=d.get('review_evidence',[])
                require(refs,'verified impact outcome has no review evidence')
                for ref in refs:
                    path=Path(ref['path']).resolve()
                    require(path.is_relative_to(Path(root).resolve()) and file_digest(path)==ref['sha256'],'impact review evidence changed')
            except (OSError,KeyError,ValueError,TypeError) as exc:
                status='under_review';reasons.append(str(exc))
        require(status in ('verified','under_review','excluded'),'unrecognized impact eligibility status')
        outcomes.append({'status':status,'change_ids':d.get('change_ids',[]),'reasons':reasons or [d.get('action','investigate')]})
    rank={'verified':0,'under_review':1,'excluded':2};worst=max(outcomes,key=lambda d:rank[d['status']])
    return {**worst,'quota_eligible':worst['status']=='verified','reviews':outcomes}


def assess(root,rows,groups,ledger=None):
    if ledger is None:ledger,_=read_ledger(root)
    key=id(ledger)
    if key not in _INDEXES:
        index=defaultdict(list)
        for d in (ledger or {}).get('decisions',[]):index[d['instance_id']].append(d)
        _INDEXES.clear();_INDEXES[key]=index
    decisions=_INDEXES[key]
    outcomes={r['instance_id']:decision_status(root,r,decisions)for r in rows}
    by_id={r['instance_id']:r for r in rows}; consequences={}
    for gid,group in groups.items():
        members=[by_id[i]for i in group.get('accepted',{})if i in by_id]
        standard=[r for r in members if r.get('sampling_split')=='eval_standard' and not r.get('task','').startswith('camera_relocalization')]
        if not standard or not any(not outcomes[r['instance_id']]['quota_eligible'] for r in standard):continue
        surviving=[r for r in standard if outcomes[r['instance_id']]['quota_eligible']]
        ev=[r for r in surviving if r['split']=='eval_standard'];tr=[r for r in surviving if r['split']=='train']
        if ev and not tr:
            for row in ev:outcomes[row['instance_id']]={'status':'under_review','quota_eligible':False,
                'reasons':['standard eval lost its required verified distinct training sibling'],'dependent_group':gid}
        if not ev and len(tr)>1:
            for row in tr:outcomes[row['instance_id']]={'status':'under_review','quota_eligible':False,
                'reasons':['multi-member training group lost eval; explicit group reassignment/replacement required'],'dependent_group':gid}
        effective={r['instance_id']:r['split']for r in members if outcomes[r['instance_id']]['quota_eligible']}
        consequences[gid]={'historical_accepted':group['accepted'],'effective_accepted':effective,
            'singleton_reassessment':not ev and len(tr)==1,'history_preserved':True}
    return outcomes,consequences


def quota_exclusions(root,db):
    ledger,_=read_ledger(root)
    if ledger is None:return set()
    decisions=ledger.get('decisions',[]);ids={d['instance_id']for d in decisions}
    if not ids:return set()
    rows=[];groups={};gids=set()
    for offset in range(0,len(ids),300):
        batch=sorted(ids)[offset:offset+300]
        query='SELECT record FROM members WHERE id IN ('+','.join('?'for _ in batch)+')'
        for (value,)in db.execute(query,batch):
            row=json.loads(value);rows.append(row)
            if row.get('group_id'):gids.add(row['group_id'])
    for gid in gids:
        result=db.execute('SELECT record FROM groups WHERE id=?',(gid,)).fetchone()
        if result:groups[gid]=json.loads(result[0])
    existing={r['instance_id']for r in rows}
    for group in groups.values():
        for row in group.get('members',[]):
            if row['instance_id']not in existing:rows.append(row);existing.add(row['instance_id'])
    outcomes,_=assess(root,rows,groups,ledger)
    return {iid for iid,result in outcomes.items()if not result['quota_eligible']}


def install_quota_view(root,db):
    """A connection-local read view; INSERTs and historical tables stay intact."""
    blocked=quota_exclusions(root,db)
    db.execute('CREATE TEMP TABLE IF NOT EXISTS impact_excluded_ids(id TEXT PRIMARY KEY)')
    db.execute('DELETE FROM impact_excluded_ids')
    db.executemany('INSERT INTO impact_excluded_ids VALUES (?)',[(i,)for i in blocked])
    db.execute('DROP VIEW IF EXISTS temp.current_members')
    db.execute('CREATE TEMP VIEW current_members AS SELECT * FROM main.members'+(' WHERE id NOT IN (SELECT id FROM impact_excluded_ids)' if blocked else ''))
    return blocked


def filtered_camera(root,accepted):
    rows=[{**r,'task':r.get('task',r.get('contract',{}).get('task'))}for r in accepted.values()];outcomes,_=assess(root,rows,{})
    return {i:r for i,r in accepted.items()if outcomes[r['instance_id']]['quota_eligible']}


def effective_group(root,row,group):
    if group is None:return group
    members=group.get('members',[]);outcomes,_=assess(root,members,{row['group_id']:group})
    if not outcomes.get(row['instance_id'],{}).get('quota_eligible',False):return None
    selected=[r for r in members if outcomes[r['instance_id']]['quota_eligible']]
    accepted={r['instance_id']:r['split']for r in selected}
    return {**group,'members':selected,'accepted':accepted,
            'deferred':[] if accepted=={row['instance_id']:'train'} else group.get('deferred',[])}


def publish(root):
    from scripts.campaign_20261110.export_release import snapshot_allocations
    from scripts.campaign_20261110.training_eligibility import classify_record,audit_eligible
    root=Path(root);rows,groups,inputs=snapshot_allocations(root);ledger,ref=read_ledger(root)
    outcomes,consequences=assess(root,rows,groups,ledger)
    counts={k:Counter()for k in ('historical','current_eligible','change_review_verified','under_review','excluded','scope_not_reconciled')}
    classes=defaultdict(Counter);tasks=defaultdict(Counter);records={};checked=set()
    for row in rows:
        split=row['split'];iid=row['instance_id'];status=outcomes[iid];eligible=status['quota_eligible'];reasons=list(status['reasons'])
        if split=='train':
            decision=classify_record(row,effective_group(root,row,groups.get(row.get('group_id'))),root=root)
            if decision['status']!='eligible':eligible=False;reasons+=decision['reasons'];status={**status,'status': 'under_review' if decision['status']=='unverified' else 'excluded'}
            else:
                try:audit_eligible(row,checked)
                except (OSError,ValueError,KeyError,TypeError)as exc:eligible=False;status={**status,'status':'under_review'};reasons.append(str(exc))
        key=(row.get('class_id',row['task']),split,row['task']);counts['historical'][split]+=1;classes[key]['historical']+=1;tasks[(row['task'],split)]['historical']+=1
        if eligible:counts['current_eligible'][split]+=1;classes[key]['current_eligible']+=1;tasks[(row['task'],split)]['current_eligible']+=1
        label='change_review_verified' if status['status']=='verified' else status['status']
        if label=='change_review_verified' and not eligible:label='excluded'
        counts[label][split]+=1;classes[key][label]+=1;tasks[(row['task'],split)][label]+=1
        records[iid]={'accepted_record_sha256':digest(clean(row)),'status':status['status'],
                      'current_eligible':eligible,'reasons':reasons,'group_id':row.get('group_id'),'split':split}
    now=datetime.now(timezone.utc).isoformat();value={'schema':'isg-current-eligibility-v1','campaign_id':'campaign_20261110',
        'observed_at':now,'ledger':ref,'status':'complete' if ledger else 'impact_inventory_unavailable',
        'review_inventory_complete':bool(ledger and ledger.get('coverage',{}).get('complete')),
        'counts':{k:dict(v)for k,v in counts.items()},'classes':[{'class_id':k[0],'split':k[1],'task':k[2],**v}for k,v in sorted(classes.items())],
        'tasks':[{'task':k[0],'split':k[1],**v}for k,v in sorted(tasks.items())],
        'groups':consequences,'records':records,'replacements':[x for d in (ledger or {}).get('decisions',[])for x in d.get('replacements',[])],
        'inputs':inputs,'history_preserved':True,'scope_gaps':(ledger or {}).get('inventory',{}).get('gaps',['impact inventory missing'] if not ledger else [])}
    destination=root/'accounting/current_eligibility.json';history=[]
    if destination.exists():
        prior=json.loads(destination.read_text());history=prior.get('history',[])[-999:]+[{'observed_at':prior['observed_at'],'eligible_counts':prior['counts']['current_eligible'],'classes':prior['classes'],'tasks':prior.get('tasks',[])}]
    value['history']=history;archive=root/'accounting/current_eligibility_history'/(str(time.time_ns())+'.json')
    write_json_atomic(archive,value);value['history_artifact']={'path':str(archive),'sha256':file_digest(archive)};write_json_atomic(destination,value)
    return {k:value[k]for k in ('observed_at','status','review_inventory_complete','counts')}


def pending_instance_blockers(root,instance_ids):
    """Known affected generated IDs cannot enter acceptance before their review."""
    ledger,_=read_ledger(root);wanted=set(instance_ids);blocked={}
    for row in (ledger or {}).get('decisions',[]):
        if row['instance_id']not in wanted:continue
        if row['status'] in ('under_review','excluded'):
            blocked[row['instance_id']]=row['status']
    return blocked


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    while True:
        try:print(json.dumps(publish(args.root)),flush=True)
        except Exception as exc:
            write_json_atomic(args.root/'accounting/current_eligibility_error.json',{'observed_at':datetime.now(timezone.utc).isoformat(),'error':repr(exc)})
            if not args.watch:raise
        if not args.watch:break
        time.sleep(120)
