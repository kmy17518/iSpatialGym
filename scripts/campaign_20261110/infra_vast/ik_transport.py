"""Coordinator half of the private campaign exact-IK exchange lifecycle."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shlex
import time
import re
import sys
import tempfile
import subprocess

_CODE = str(Path(__file__).resolve().parents[3])
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

CAMPAIGN='campaign_20261110'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,sort_keys=True)+'\n');tmp.replace(path)


NATIVE_PYTHON = Path('/vision/u/minyeong/miniconda3/envs/ispatialgym-vectorized-envs/bin/python')


def verify_native_manifest(job, manifest_path, environment, helper_code):
    """AST semantic hashes must be checked by the actual native Python version.

    Only pure read-only compatibility code runs here; no simulator or cache is
    loaded. The immutable manifest is compared, never rebuilt or weakened.
    """
    if not NATIVE_PYTHON.is_file():
        raise ValueError('native interpreter unavailable for exact-IK verification')
    request = {'helper': str(helper_code), 'manifest': str(manifest_path),
               'code': environment['ISGE_CODE'],
               'config': environment['ISPATIALGYM_CAMPAIGN_CONFIG'],
               'environment': environment}
    code = """import json,sys,pathlib,hashlib
r=json.load(sys.stdin);sys.path.insert(0,r['helper'])
from scripts.campaign_20261110.ik_compatibility import verify_running
p=pathlib.Path(r['manifest']);m=json.loads(p.read_text())
verify_running(m,r['code'],r['config'],r['environment'])
print(json.dumps({'verified':True,'compatibility_id':m['compatibility_sha256'],
 'manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
 'native_interpreter':sys.executable,'python_version':sys.version}))
"""
    result = subprocess.run([str(NATIVE_PYTHON), '-c', code], input=json.dumps(request),
                            text=True, capture_output=True, timeout=120)
    if result.returncode:
        raise ValueError('native exact-IK compatibility failed: ' + result.stderr[-1200:])
    proof = json.loads(result.stdout)
    if proof.get('verified') is not True or proof.get('manifest_sha256') != sha(manifest_path):
        raise ValueError('native exact-IK verification receipt differs')
    atomic(Path(job)/'exact_ik_native_verification.json', proof)
    return proof


def prepare(root,job,env,s3,bucket,prefix,helper_code):
    """Only explicitly instrumented new jobs opt in; old native jobs stay intact."""
    manifest_path=env.get('ISPATIALGYM_EXACT_IK_MANIFEST')
    if not manifest_path:return None
    root,job,manifest_path=Path(root).resolve(),Path(job).resolve(),Path(manifest_path).resolve()
    if root.name!=CAMPAIGN or not job.is_relative_to(root/'out') or not manifest_path.is_relative_to(root):raise ValueError('foreign IK exchange input')
    from scripts.campaign_20261110.ik_compatibility import validate_manifest
    from scripts.campaign_20261110.ik_exchange import pack_delta
    manifest=validate_manifest(json.loads(manifest_path.read_text()))
    effective={**manifest.get('provenance',{}).get('environment',{}),**env}
    verify_native_manifest(job,manifest_path,effective,helper_code)
    native=Path(env['ISGE_CODE'])/'ispatialgym/sampling/v2/ik.py'
    if 'register_query' not in native.read_text():raise ValueError('native runtime has no validated exact witness writer')
    exchange=root/'ik_exchange';identity=manifest['compatibility_sha256'];ms=sha(manifest_path)
    canonical=exchange/'manifests'/(ms+'.json');canonical.parent.mkdir(parents=True,exist_ok=True)
    # Preserve exact bytes atomically, including concurrent same-manifest staging.
    with tempfile.NamedTemporaryFile(dir=canonical.parent,delete=False) as stream:
        stream.write(manifest_path.read_bytes());temporary=Path(stream.name)
    temporary.replace(canonical)
    seed_dir=exchange/'seeds';seed_dir.mkdir(parents=True,exist_ok=True)
    temp=seed_dir/(hashlib.sha256(str(job).encode()).hexdigest()+'.pending.json')
    packed=pack_delta(exchange/'durable',manifest,temp,max_entries=256,max_bytes=8*1024*1024,deadline_seconds=5,source='campaign_durable_exchange')
    seed=None
    if packed['entry_count']:
        immutable=seed_dir/(packed['sha256']+'.json');temp.replace(immutable)
        key='ik_exchange/seeds/'+packed['sha256']+'.json';s3.upload_file(str(immutable),bucket,prefix+'/'+key)
        seed={'key':key,'sha256':packed['sha256'],'bytes':packed['bytes'],'entries':packed['entry_count'],'branches':packed['branch_count']}
    else:temp.unlink(missing_ok=True)
    spec={'campaign_id':CAMPAIGN,'compatibility_id':identity,'manifest_path':str(manifest_path),'manifest_sha256':ms,
          'helper_code':str(helper_code),'seed':seed,'original_cache_scope':env.get('ISGF_SAMPLING_CACHE'),
          'local_cache':'/work/campaign_20261110/exact_ik/cache','periodic_seconds':180,'bounded_seed_bytes':8*1024*1024}
    atomic(job/'exact_ik_transport.json',spec)
    return spec


def reconcile(root,s3,bucket,prefix,*,max_descriptors=8,deadline_seconds=15):
    """Incrementally merge uploaded updates including richer existing keys.

    Receipts keep retries idempotent. Failed/deferred bundles are revisited on
    subsequent bounded scans, and original immutable objects are never deleted.
    """
    from scripts.campaign_20261110.ik_exchange import ingest_bundle
    root=Path(root).resolve();assert root.name==CAMPAIGN
    base=root/'ik_exchange';state_path=base/'coordinator_cursor.json'
    state=json.loads(state_path.read_text())if state_path.exists()else{}
    args={'Bucket':bucket,'Prefix':prefix+'/ik_exchange/published/','MaxKeys':max_descriptors*2}
    if state.get('token'):args['ContinuationToken']=state['token']
    start=time.monotonic();page=s3.list_objects_v2(**args);records=[]
    for item in page.get('Contents',[]):
        if time.monotonic()-start>deadline_seconds:break
        key=item['Key']
        if not key.endswith('.json.receipt.json'):continue
        if int(item.get('Size',0))>32768:continue
        ident=hashlib.sha256(key.encode()).hexdigest();receipt=base/'ingest_receipts'/(ident+'.json')
        if receipt.exists()and json.loads(receipt.read_text()).get('complete'):continue
        try:
            d=json.loads(s3.get_object(Bucket=bucket,Key=key)['Body'].read(32769))
            if d.get('campaign_id')!=CAMPAIGN:raise ValueError('foreign publication')
            if any(not re.fullmatch('[0-9a-f]{64}',str(d.get(k,''))) for k in ('sha256','manifest_sha256','compatibility_id')):raise ValueError('malformed publication hash')
            if any(not str(d.get(k,'')).isdigit() for k in ('instance','job_id')):raise ValueError('malformed publication identity')
            expected=prefix+'/ik_exchange/published/'+str(d['instance'])+'/'+str(d['job_id'])+'/'+d['sha256']+'.json'
            if key!=expected+'.receipt.json' or prefix+'/'+d['bundle_key']!=expected:raise ValueError('publication path identity mismatch')
            manifest=base/'manifests'/(d['manifest_sha256']+'.json')
            if sha(manifest)!=d['manifest_sha256']:raise ValueError('unregistered manifest identity')
            if not 0<int(d['bytes'])<=20*1024*1024:raise ValueError('oversize publication')
            temp=base/'incoming'/(d['sha256']+'.json');temp.parent.mkdir(parents=True,exist_ok=True)
            body=s3.get_object(Bucket=bucket,Key=expected)['Body'].read(20*1024*1024+1)
            if len(body)!=d['bytes']:raise ValueError('publication size differs')
            temp.write_bytes(body)
            report=ingest_bundle(temp,base/'durable',manifest,receipt=receipt,expected_sha256=d['sha256'],deadline_seconds=5,max_entries=256)
            ack={'campaign_id':CAMPAIGN,'bundle_sha256':d['sha256'],'compatibility_id':d['compatibility_id'],
                 'complete':report['complete'],'entries':report['entries'],'branches':report['branches'],
                 'durable_receipt':report['durable_receipt'],'errors':report['errors'],'deferred':report['deferred'],'observed_at':time.time()}
            s3.put_object(Bucket=bucket,Key=prefix+'/ik_exchange/acks/'+ident+'.json',Body=json.dumps(ack).encode())
            records.append(ack)
        except Exception as exc:
            failure={'campaign_id':CAMPAIGN,'key':key,'complete':False,'error':type(exc).__name__+': '+str(exc)[:300],'observed_at':time.time()}
            atomic(receipt,failure);records.append(failure)
    # Restart scanning after the last page; prior complete receipts make replay cheap.
    atomic(state_path,{'token':page.get('NextContinuationToken'),'observed_at':time.time()})
    report={'campaign_id':CAMPAIGN,'observed_at':time.time(),'records':records,'elapsed_seconds':time.monotonic()-start}
    atomic(base/'coordinator_status.json',report)
    return report
