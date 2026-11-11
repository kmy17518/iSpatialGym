"""Bounded incremental transport for checked campaign exact-IK witnesses.

Bundles are typed JSON, never extractable archives. Only complete successful
witness entries with query/checksum/compatibility evidence are transported.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import tempfile
import time
from scripts.campaign_20261110.ik_compatibility import CAMPAIGN,digest,file_hash,validate_manifest
from ispatialgym.sampling.exact_exchange import KEY,atomic,read_entry,validate_entry,install

BUNDLE_SCHEMA='isg-exact-ik-bundle-v1'

def load_manifest(manifest):
 return validate_manifest(json.loads(Path(manifest).read_text()) if isinstance(manifest,(str,Path)) else manifest)

def directory(root,manifest):
 p=Path(root)
 if CAMPAIGN not in str(p.resolve()):raise ValueError('cache/exchange is outside campaign namespace')
 return p/manifest['compatibility_sha256']

def pack_delta(cache,manifest,bundle,*,cursor=None,max_entries=512,max_bytes=16*1024*1024,deadline_seconds=10.,source=None):
 """Return new cursor; caller persists it ONLY after durable upload succeeds."""
 t=time.monotonic();manifest=load_manifest(manifest);base=directory(cache,manifest)
 old=cursor or {};updated=dict(old);entries=[];failures=[];bytes_used=0;complete=True
 for p in sorted(base.glob('exact_ik_v3-*.json')):
  if len(entries)>=max_entries or time.monotonic()-t>=deadline_seconds:complete=False;break
  key=p.stem
  try:
   if not KEY.fullmatch(key):raise ValueError('unsafe entry filename')
   stat=p.stat();signature=[stat.st_size,stat.st_mtime_ns]
   if old.get(key,{}).get('stat')==signature:continue
   value=read_entry(p,key,manifest);sha=digest(value)
   if old.get(key,{}).get('sha256')==sha:updated[key]={'stat':signature,'sha256':sha};continue
   size=len(json.dumps(value,allow_nan=False).encode())
   if bytes_used+size>max_bytes:complete=False;break
   entries.append({'key':key,'sha256':sha,'value':value});bytes_used+=size
   updated[key]={'stat':signature,'sha256':sha}
  except (OSError,ValueError,KeyError,TypeError)as exc:
   failures.append({'key':key,'error':str(exc)})
 payload={'schema':BUNDLE_SCHEMA,'campaign_id':CAMPAIGN,'compatibility_id':manifest['compatibility_sha256'],
   'observed_at':datetime.now(timezone.utc).isoformat(),'source':source,'entries':entries}
 payload['bundle_sha256']=digest(payload);atomic(bundle,payload)
 return {'bundle_path':str(bundle),'sha256':file_hash(bundle),'content_sha256':payload['bundle_sha256'],
   'entry_count':len(entries),'branch_count':sum(1+len(e['value'].get('alts',[])) for e in entries),
   'bytes':Path(bundle).stat().st_size,'compatibility_id':manifest['compatibility_sha256'],'cursor':updated,
   'elapsed_seconds':time.monotonic()-t,'scan_complete':complete,'failures':failures[:100],
   'cursor_commit_rule':'Persist only after successful durable upload/ingest acknowledgement.'}

def ingest_bundle(bundle,exchange,manifest,*,receipt=None,expected_sha256=None,max_entries=512,max_bytes=20*1024*1024,deadline_seconds=20.):
 t=time.monotonic();manifest=load_manifest(manifest);p=Path(bundle)
 if p.is_symlink() or p.stat().st_size>max_bytes:raise ValueError('unsafe/oversize bundle')
 if expected_sha256 is not None and file_hash(p)!=expected_sha256:raise ValueError('transport checksum mismatch')
 value=json.loads(p.read_text())
 if value.get('schema')!=BUNDLE_SCHEMA or value.get('campaign_id')!=CAMPAIGN:raise ValueError('foreign bundle')
 if value.get('compatibility_id')!=manifest['compatibility_sha256']:raise ValueError('bundle compatibility mismatch')
 if value.get('bundle_sha256')!=digest({k:v for k,v in value.items() if k!='bundle_sha256'}):raise ValueError('bundle checksum mismatch')
 entries=value.get('entries')
 if not isinstance(entries,list) or len(entries)>max_entries:raise ValueError('entry limit exceeded')
 keys=set()
 for entry in entries:
  key=entry['key']
  if not KEY.fullmatch(key) or key in keys:raise ValueError('unsafe/duplicate bundle key')
  keys.add(key)
  if digest(entry['value'])!=entry['sha256']:raise ValueError('entry checksum mismatch')
  validate_entry(entry['value'],key,manifest)
 base=directory(exchange,manifest);results=[];errors=[];deferred=[]
 for i,entry in enumerate(entries):
  if time.monotonic()-t>=deadline_seconds:deferred=[e['key']for e in entries[i:]];break
  try:results.append({'key':entry['key'],**install(base,entry['value'],entry['key'],manifest)})
  except (OSError,ValueError,KeyError,TypeError,TimeoutError)as exc:errors.append({'key':entry['key'],'error':str(exc)})
 report={'campaign_id':CAMPAIGN,'schema':'isg-exact-ik-transfer-receipt-v1','observed_at':datetime.now(timezone.utc).isoformat(),
  'compatibility_id':manifest['compatibility_sha256'],'source':value.get('source'),
  'bundle':{'path':str(p),'sha256':file_hash(p),'content_sha256':value['bundle_sha256']},
  'destination':str(base),'entries':len(results),'branches':sum(r['branches']for r in results),
  'changed_entries':sum(r['changed']for r in results),'bytes':p.stat().st_size,
  'elapsed_seconds':time.monotonic()-t,'results':results,'errors':errors,'deferred':deferred,
  'complete':not errors and not deferred,'originals_preserved':True,
  'semantics':'Installed candidates only. A verified IK hit requires unchanged live FK/collision/support checks.'}
 durable=Path(exchange)/'receipts'/(value['bundle_sha256']+'_'+str(time.time_ns())+'.json');atomic(durable,report)
 report['durable_receipt']={'path':str(durable),'sha256':file_hash(durable)}
 if receipt:atomic(receipt,report)
 return report

def import_entries(exchange,cache,manifest,*,cursor=None,receipt=None,**bounds):
 with tempfile.TemporaryDirectory(prefix='campaign_20261110_ik_')as tmp:
  exported=pack_delta(exchange,manifest,Path(tmp)/'bundle.json',cursor=cursor,source='durable_campaign_exchange',**bounds)
  imported=ingest_bundle(exported['bundle_path'],cache,manifest,receipt=receipt,expected_sha256=exported['sha256'],
    max_entries=bounds.get('max_entries',512),max_bytes=max(exported['bytes'],20*1024*1024),deadline_seconds=bounds.get('deadline_seconds',20.))
 return {**imported,'cursor':exported['cursor']if imported['complete']else cursor,'export_failures':exported['failures']}

def publish(cache,exchange,manifest,*,cursor=None,receipt=None,source=None,**bounds):
 with tempfile.TemporaryDirectory(prefix='campaign_20261110_ik_')as tmp:
  exported=pack_delta(cache,manifest,Path(tmp)/'bundle.json',cursor=cursor,source=source,**bounds)
  imported=ingest_bundle(exported['bundle_path'],exchange,manifest,receipt=receipt,expected_sha256=exported['sha256'],
    max_entries=bounds.get('max_entries',512),max_bytes=max(exported['bytes'],20*1024*1024),deadline_seconds=bounds.get('deadline_seconds',20.))
 return {**imported,'cursor':exported['cursor']if imported['complete']else cursor,'export_failures':exported['failures']}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['pack','ingest','import','publish'])
 for name in ['manifest','cache','bundle','exchange','receipt','cursor','source','expected-sha256']:p.add_argument('--'+name)
 p.add_argument('--max-entries',type=int,default=512);p.add_argument('--max-bytes',type=int,default=16*1024*1024);p.add_argument('--deadline-seconds',type=float,default=10.)
 a=p.parse_args();bounds=dict(max_entries=a.max_entries,max_bytes=a.max_bytes,deadline_seconds=a.deadline_seconds)
 cursor=json.loads(Path(a.cursor).read_text()) if a.cursor and Path(a.cursor).exists() else None
 if a.action=='pack':r=pack_delta(a.cache,a.manifest,a.bundle,cursor=cursor,source=a.source,**bounds)
 elif a.action=='ingest':r=ingest_bundle(a.bundle,a.exchange or a.cache,a.manifest,receipt=a.receipt,expected_sha256=a.expected_sha256,**bounds)
 elif a.action=='import':r=import_entries(a.exchange,a.cache,a.manifest,cursor=cursor,receipt=a.receipt,**bounds)
 else:r=publish(a.cache,a.exchange,a.manifest,cursor=cursor,receipt=a.receipt,source=a.source,**bounds)
 if a.receipt:atomic(a.receipt,r)
 print(json.dumps(r))

if __name__=='__main__':main()
