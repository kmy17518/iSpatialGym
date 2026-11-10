#!/usr/bin/env python3
"""Read-only accepted-use snapshot before adding a reserved generated model."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sqlite3


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.campaign_root.resolve();out=a.output.resolve()
    if out.exists() or not out.is_relative_to(root):raise ValueError('new private audit required')
    key='classroom_mock_up/isgm10';ready_path=root/'manifests/generated_project_readiness_r2.json'
    obj=json.loads(ready_path.read_text())['objects'][key];asset_hash=obj['asset_sha256']
    if not obj['ready'] or sha(Path(obj['usd']))!=asset_hash:raise ValueError('generated model readiness/bytes mismatch')
    db=root/'dataset/allocations.sqlite';conn=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True)
    conn.execute('PRAGMA query_only=ON');conn.execute('BEGIN')
    tables={name:conn.execute('SELECT * FROM '+name+' ORDER BY 1').fetchall() for name in ['metadata','groups','members','assets']}
    conn.rollback();conn.close()
    hits={name:[row for row in rows if 'isgm10' in json.dumps(row) or asset_hash in json.dumps(row)] for name,rows in tables.items() if name!='metadata'}
    cameras=[]
    for filename in ['camera_quota.json','camera_obstructed_quota.json','camera_category_quota.json']:
        path=root/'dataset'/filename
        if not path.exists():cameras.append({'path':str(path),'exists':False});continue
        source_bytes=path.read_bytes();document=json.loads(source_bytes);rows=document.get('accepted',{});rows=list(rows.values()) if isinstance(rows,dict) else rows
        matches=[];tro_evidence=[]
        for row in rows:
            if 'isgm10' in json.dumps(row) or asset_hash in json.dumps(row):matches.append({'row':row})
            tro=(row.get('artifacts',{}).get('tro') or {}).get('path')
            if tro:
                tp=Path(tro);text=tp.read_text();tro_evidence.append({'path':str(tp),'sha256':sha(tp)})
                if 'isgm10' in text or asset_hash in text:matches.append({'tro':str(tp)})
        cameras.append({'path':str(path),'exists':True,'sha256':hashlib.sha256(source_bytes).hexdigest(),'accepted_count':len(rows),'matches':matches,'tro_evidence':tro_evidence})
    configs=[];config_hits=[]
    for path in sorted((root/'groups').glob('**/campaign_config.json')):
        text=path.read_text();configs.append({'path':str(path),'sha256':sha(path)})
        if 'isgm10' in text or asset_hash in text:config_hits.append(str(path))
    passed=not any(hits.values()) and not any(c.get('matches') for c in cameras) and not config_hits
    result={'schema':'campaign-reservation-extension-use-audit-v1','audited_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'model_key':key,'asset_sha256':asset_hash,'ready':True,'readiness_source':obj['readiness_source'],'readiness_registry_sha256':sha(ready_path),
        'db_path':str(db),'db_snapshot_counts':{k:len(v) for k,v in tables.items()},'db_snapshot_sha256':digest(tables),'db_matches':hits,
        'camera_states':cameras,'group_configs':configs,'group_config_matches':config_hits,'no_accepted_or_group_config_use':passed,
        'compute_owner_active_job_report':{'raw_physical_prebuild_only':'18421518_9','manifest':str(root/'room_materialized_v2/generated_mental_floating_jobs.json'),
            'message':'Compute owner confirms no isgm10 room sampling, annotation, production or accepted generated mental instances; raw physical library may serve new eval holdout if identical.'},
        'scope':'Read-only snapshot; allocator must independently recheck under migration lock before adopting new policy.'}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'passed':passed,'counts':result['db_snapshot_counts'],'camera_accepted':sum(x.get('accepted_count',0) for x in cameras),'group_configs':len(configs)}))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
