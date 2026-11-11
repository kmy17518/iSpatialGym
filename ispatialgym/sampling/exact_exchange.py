"""Local-only, validated exact-IK witness sharing for campaign_20261110.

No remote requests occur on the query path. Exchanged witnesses remain candidates
for IKChecker's unchanged current-world checks. Other sampling caches stay local
and untouched. Missing/invalid inputs fall back to the ordinary cache/solver.
"""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import tempfile
import socket
import uuid
import time

from scripts.campaign_20261110.ik_compatibility import CAMPAIGN,digest,validate_manifest,verify_running,file_hash

LOG=logging.getLogger(__name__)
ENTRY_SCHEMA='isg-exact-ik-entry-v1'
KEY=re.compile(r'^exact_ik_v3-[0-9a-f]{24}$')
MAX_ENTRY_BYTES=2*1024*1024
_QUERIES={};_CONTEXT={};_COUNTERS={};_LAST_FLUSH=0.


def canonical_query(value):
 if isinstance(value,dict):return {str(k):canonical_query(v) for k,v in sorted(value.items())}
 if isinstance(value,(tuple,list)):return [canonical_query(x) for x in value]
 if hasattr(value,'tolist'):return canonical_query(value.tolist())
 if isinstance(value,float):return round(value,4)
 return value

def query_key(payload):
 raw=json.dumps({'schema':1,'kind':'exact_ik_v3','payload':payload},sort_keys=True,allow_nan=False)
 return 'exact_ik_v3-'+hashlib.sha256(raw.encode()).hexdigest()[:24]

def validate_entry(value,key,manifest):
 validate_manifest(manifest)
 if not KEY.fullmatch(key) or not isinstance(value,dict) or value.get('ok') is not True:raise ValueError('not a successful exact witness')
 m=value.get('_exchange',{})
 if m.get('schema')!=ENTRY_SCHEMA or m.get('campaign_id')!=CAMPAIGN:raise ValueError('missing/foreign exact witness provenance')
 if m.get('compatibility_sha256')!=manifest['compatibility_sha256']:raise ValueError('incompatible witness')
 if query_key(m.get('query'))!=key or m.get('key')!=key:raise ValueError('query key mismatch')
 if value.get('_checksum')!=digest({k:v for k,v in value.items() if k!='_checksum'}):raise ValueError('witness checksum mismatch')
 dof=m.get('dof');idx=m.get('arm_indices');poses=m.get('query',{}).get('poses',[])
 if not isinstance(dof,int) or not 1<=dof<=128 or not isinstance(idx,list) or not idx or len(set(idx))!=len(idx) or any(type(i)is not int or i<0 or i>=dof for i in idx):raise ValueError('invalid kinematic dimensions')
 if not 1<=len(poses)<=128:raise ValueError('invalid query pose count')
 for pose in poses:
  if not isinstance(pose,list) or len(pose)!=7 or any(type(x)not in (float,int) or not math.isfinite(x) for x in pose):raise ValueError('malformed query pose')
 if m['query'].get('arm') not in ('left','right'):raise ValueError('missing arm identity')
 chains=[value.get('qs')]+value.get('alts',[])
 if not 1<=len(chains)<=manifest['compatible_identity']['branch_limit']:raise ValueError('branch limit exceeded')
 for chain in chains:
  if not isinstance(chain,list) or len(chain)!=len(poses):raise ValueError('partial witness chain')
  for q in chain:
   if not isinstance(q,list) or len(q)!=dof or any(type(x)not in (float,int) or not math.isfinite(x) for x in q):raise ValueError('malformed joint witness')
 if value.get('q')!=chains[0][-1]:raise ValueError('primary witness differs from chain')
 return value

def seal(value):
 value={k:v for k,v in value.items() if k!='_checksum'}
 return {**value,'_checksum':digest(value)}

def merge_entries(old,new,key,manifest):
 validate_entry(new,key,manifest)
 if old is None:return new
 validate_entry(old,key,manifest)
 if old['_exchange']['query']!=new['_exchange']['query'] or old['_exchange']['arm_indices']!=new['_exchange']['arm_indices'] or old['_exchange']['dof']!=new['_exchange']['dof']:raise ValueError('same-key identity conflict')
 idx=new['_exchange']['arm_indices'];threshold=manifest['compatible_identity']['branch_min_rad'];limit=manifest['compatible_identity']['branch_limit']
 chains=[old['qs']]+old.get('alts',[])
 for candidate in [new['qs']]+new.get('alts',[]):
  if all(max(abs(candidate[-1][i]-c[-1][i]) for i in idx)>threshold for c in chains):chains.append(candidate)
 chosen=[chains[0]]+chains[1:][-(limit-1):] if limit>1 else chains[:1]
 provenance=sorted(set(old['_exchange'].get('publishers',[])+new['_exchange'].get('publishers',[])))[:64]
 return seal({**old,'qs':chosen[0],'q':chosen[0][-1],'alts':chosen[1:],'_exchange':{**old['_exchange'],'publishers':provenance}})

@contextmanager
def file_lock(path,timeout=2.):
 # Atomic mkdir works on campaign NFS, where flock returns EIO. A lock whose
 # owner cannot be proved gone is left for diagnosis, never stolen by age.
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 end=time.monotonic()+timeout;token=uuid.uuid4().hex
 while True:
  try:path.mkdir();break
  except FileExistsError:
   if time.monotonic()>=end:raise TimeoutError('bounded cache merge lock timed out: '+str(path))
   time.sleep(.02)
 try:
  (path/'owner.json').write_text(json.dumps({'token':token,'pid':os.getpid(),'host':socket.gethostname(),'created_at':time.time()}))
  yield
 finally:
  try:
   owner=json.loads((path/'owner.json').read_text())
   if owner.get('token')==token:(path/'owner.json').unlink();path.rmdir()
  except OSError:pass

def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=path.parent)
 try:
  with os.fdopen(fd,'w')as f:json.dump(value,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  try:os.unlink(tmp)
  except FileNotFoundError:pass

def read_entry(path,key,manifest):
 p=Path(path)
 if p.is_symlink() or p.stat().st_size>MAX_ENTRY_BYTES:raise ValueError('unsafe/oversize witness entry')
 return validate_entry(json.loads(p.read_text()),key,manifest)

def install(directory,value,key,manifest):
 directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);path=directory/(key+'.json')
 with file_lock(directory/('.'+key+'.lock')):
  try:old=read_entry(path,key,manifest)
  except FileNotFoundError:old=None
  # A bad old entry is preserved for diagnosis; never silently overwrite it.
  merged=merge_entries(old,value,key,manifest)
  if old!=merged:atomic(path,merged)
 return {'changed':old!=merged,'branches':1+len(merged.get('alts',[])),'sha256':file_hash(path)}

def context():
 path=os.environ.get('ISPATIALGYM_EXACT_IK_MANIFEST');root=os.environ.get('ISPATIALGYM_EXACT_IK_LOCAL_CACHE')
 if not path or not root:return None
 key=(path,root,os.environ.get('ISPATIALGYM_CAMPAIGN_CONFIG'))
 if key not in _CONTEXT:
  try:
   manifest=json.loads(Path(path).read_text());code=Path(__file__).resolve().parents[2]
   verify_running(manifest,code,os.environ['ISPATIALGYM_CAMPAIGN_CONFIG'],dict(os.environ))
   if CAMPAIGN not in str(Path(root).resolve()):raise ValueError('local cache outside isolated campaign namespace')
   directory=Path(root)/manifest['compatibility_sha256'];directory.mkdir(parents=True,exist_ok=True)
   _CONTEXT[key]=(manifest,directory)
  except (OSError,ValueError,KeyError,TypeError)as exc:
   LOG.warning('Exact witness sharing unavailable; ordinary fresh operation: %s',exc);_CONTEXT[key]=None
 return _CONTEXT[key]

def register_query(key,payload,indices,dof):
 if context() is None:return
 try:
  # Match the existing key serializer exactly, including flattened tensors and
  # NumPy scalar handling. Transport helpers remain simulator-free: this import
  # runs only on the sampler's query path, where cache is already loaded.
  from ispatialgym.sampling.cache import _plain,_json_default
  payload=json.loads(json.dumps(_plain(payload),sort_keys=True,default=_json_default,allow_nan=False))
  if query_key(payload)!=key:raise ValueError('exact cache key serialization differs')
  if len(_QUERIES)>=8192:_QUERIES.pop(next(iter(_QUERIES)))
  _QUERIES[key]={'query':payload,'arm_indices':list(indices),'dof':int(dof)}
 except (ValueError,TypeError,KeyError)as exc:
  _QUERIES.pop(key,None);record_event('unshareable_query')
  LOG.warning('Exact witness sharing skipped for this query; ordinary cache/solver retained: %s',exc)

def read_shared(key):
 c=context()
 if c is None or key not in _QUERIES:return None
 manifest,directory=c
 try:
  value=read_entry(directory/(key+'.json'),key,manifest)
  if any(value['_exchange'][k]!=v for k,v in _QUERIES[key].items()):raise ValueError('current query differs')
  return value
 except FileNotFoundError:return None
 except (OSError,ValueError,KeyError,TypeError)as exc:
  record_event('invalid_shared_entry');LOG.warning('Ignoring shared exact witness %s: %s',key,exc);return None

def write_shared(key,value):
 c=context()
 if c is None or key not in _QUERIES or value.get('ok')is not True:return
 manifest,directory=c
 try:
  entry=seal({**canonical_query(value),'_exchange':{'schema':ENTRY_SCHEMA,'campaign_id':CAMPAIGN,
   'key':key,'compatibility_sha256':manifest['compatibility_sha256'],**_QUERIES[key],
   'publishers':[os.environ.get('ISPATIALGYM_EXACT_IK_PRODUCER','unattributed')]}})
  result=install(directory,entry,key,manifest)
  record_event('shared_entries_updated' if result['changed']else'shared_entries_unchanged')
 except (OSError,ValueError,KeyError,TypeError,TimeoutError)as exc:
  record_event('shared_write_failed');LOG.warning('Exact witness publication deferred: %s',exc)

def record_event(name,n=1,force=False):
 global _LAST_FLUSH
 _COUNTERS[name]=_COUNTERS.get(name,0)+n
 path=os.environ.get('ISPATIALGYM_EXACT_IK_METRICS')
 if not path:return
 now=time.monotonic()
 if not force and now-_LAST_FLUSH<10:return
 _LAST_FLUSH=now
 try:
  c=context();atomic(Path(path)/(str(os.getpid())+'.json'),{'campaign_id':CAMPAIGN,
    'observed_at':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),
    'producer':os.environ.get('ISPATIALGYM_EXACT_IK_PRODUCER'),
    'compatibility_sha256':c[0]['compatibility_sha256']if c else None,
    'local_cache':str(c[1])if c else None,'counters':dict(_COUNTERS),
    'semantics':'verified_hit follows unchanged current-world checks; fresh_query counts solver-path queries, not individual solver calls'})
 except (OSError,ValueError):pass

import atexit
atexit.register(lambda: record_event("final_flush",0,force=True))
