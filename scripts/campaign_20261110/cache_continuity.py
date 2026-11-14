"""Versioned cache transport for persistent eval/training workers.

Transport, merge, publication, consumption and native hit evidence are separate.
Only typed successful exact IK entries are shared. Legacy entries are preserved
without being promoted. Original libraries and overlays are never overwritten.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import time

from scripts.campaign_20261110 import ik_compatibility as compat, ik_exchange as ik
from scripts.campaign_20261110.infra_vast.worker.warm_cache import Store, sha, write, canonical
from ispatialgym.sampling.exact_exchange import file_lock

ENV = 'ISPATIALGYM_CACHE_CONTINUITY'
SCHEMA = 'isg-cache-continuity-v1'


def read(p): return json.loads(Path(p).read_text())
def clean(s):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', s): raise ValueError('unsafe cache owner')
    return s

def put(store, path, suffix='.json'):
    h=sha(path); key='objects/'+h+suffix
    store.put_verified(path,key)
    return {'key':key,'sha256':h,'bytes':Path(path).stat().st_size}


def get(store, reference, dest):
    if reference['key'] != 'objects/'+reference['sha256']+Path(reference['key']).suffix:
        raise ValueError('not a content addressed object')
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        tmp=dest.with_name(dest.name+'.tmp.'+str(os.getpid()));store.get(reference['key'],tmp)
        if sha(tmp)!=reference['sha256']:raise ValueError('cache transfer checksum mismatch')
        tmp.replace(dest)
    if sha(dest)!=reference['sha256']:raise ValueError('durable cache checksum mismatch')
    return dest


def overlay_chunks(root, cursor, destination):
    """Export disjoint complete lines. Never re-merge a growing shard's prefix."""
    out=[];updated=dict(cursor);destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    for path in sorted(Path(root).rglob('*.jsonl')):
        if path.is_symlink():continue
        name=str(path.resolve());offset=int(cursor.get(name,0));size=path.stat().st_size
        if size<offset:raise ValueError('overlay shard truncated')
        with path.open('rb')as f:f.seek(offset);raw=f.read(8*1024*1024)
        end=raw.rfind(b'\n')+1
        if not end:continue
        raw=raw[:end]
        for line in raw.splitlines():
            value=json.loads(line)
            if not re.fullmatch(r'[A-Za-z0-9_.-]+',str(value.get('map_key',''))):raise ValueError('unsafe overlay map identity')
        key=hashlib.sha256(canonical([name,offset,offset+end])).hexdigest()
        chunk=destination/(key+'.jsonl');chunk.write_bytes(canonical({'transport_source':name,'start':offset,'end':offset+end})+b'\n'+raw)
        out.append({'path':str(chunk),'source':name,'start':offset,'end':offset+end})
        updated[name]=offset+end
    return out,updated


def prepare(settings_path, job):
    settings=read(settings_path);job=Path(job);root=job/'cache_continuity';root.mkdir(parents=True,exist_ok=True)
    store=Store(settings['store']);code=Path(__file__).resolve().parents[2]
    manifest=compat.build_manifest(code,os.environ['ISPATIALGYM_CAMPAIGN_CONFIG'],dict(os.environ),
                                  read(settings['robot_reference_manifest'])['provenance']['robot_refs'])
    mp=root/'compatibility.json';write(mp,manifest);cid=manifest['compatibility_sha256'];loaded=[]
    # Latest pointer is written only after all immutable version objects verify.
    pointer='published/'+cid+'/latest.json'
    with tempfile.TemporaryDirectory(dir=root)as tmp:
        try: store.get(pointer,Path(tmp)/'latest.json')
        except (OSError,RuntimeError,subprocess.CalledProcessError) as exc:
            # Missing first seed is allowed; other errors remain visible.
            write(root/'seed_unavailable.json',{'error':type(exc).__name__,'at':time.time()})
        else:
            version=read(Path(tmp)/'latest.json')
            if version['schema']!=SCHEMA or version['compatibility_id']!=cid:raise ValueError('foreign cache version')
            for ref in version.get('ik_bundles',[]):
                bundle=get(store,ref,Path(tmp)/ref['sha256'])
                result=ik.ingest_bundle(bundle,root/'exact',manifest,expected_sha256=ref['sha256'],deadline_seconds=60,max_entries=4096,max_bytes=128*1024*1024)
                if not result['complete']:raise ValueError('incomplete cache seed import')
                loaded.append({'bundle':ref,'entries':result['entries'],'receipt':result['durable_receipt']})
            if version.get('overlay_files'):
                for name,ref in version['overlay_files'].items():
                    parts=Path(name)
                    if parts.is_absolute()or'..'in parts.parts:raise ValueError('unsafe overlay path')
                    get(store,ref,root/'overlay'/parts)
                os.environ['ISPATIALGYM_MAP_OVERLAY']=str(root/'overlay')
                os.environ['ISPATIALGYM_MAP_OVERLAY_RELEASE']=str(version['overlay_release'])
            write(root/'consumed_version.json',{'version':version,'loaded':loaded,'at':time.time(),'actual_native_hits':'see exact_ik_metrics'})
    context={'schema':SCHEMA,'owner':clean(settings['owner']),'settings_path':str(Path(settings_path).resolve()),
             'appdata':os.environ.get('OMNIGIBSON_APPDATA_PATH'),'collectors':settings.get('collectors',2),
             'job':str(job.resolve()),'manifest':str(mp),'compatibility_id':cid,'overlay_shards':os.environ.get('ISPATIALGYM_MAP_OVERLAY_SHARDS',str(job/'map_overlay')),
             'base_overlay':os.environ.get('ISPATIALGYM_MAP_OVERLAY'),'base_overlay_release':os.environ.get('ISPATIALGYM_MAP_OVERLAY_RELEASE','0'),
             'legacy_roots':{k:os.environ[k]for k in('ISPATIALGYM_SAMPLING_CACHE','ISPATIALGYM_CACHE_DIR','ISPATIALGYM_REACH_BANDS','ISPATIALGYM_REACH_VOLUME','ISPATIALGYM_HELD_HOLD_CACHE')if os.environ.get(k)}}
    write(root/'context.json',context)
    env={'ISPATIALGYM_EXACT_IK_MANIFEST':str(mp),'ISPATIALGYM_EXACT_IK_LOCAL_CACHE':str(root/'exact'),
         'ISPATIALGYM_EXACT_IK_PRODUCER':settings['owner'],'ISPATIALGYM_EXACT_IK_METRICS':str(job/'exact_ik_metrics')}
    for k in('ISPATIALGYM_MAP_OVERLAY','ISPATIALGYM_MAP_OVERLAY_RELEASE'):
        if k in os.environ:env[k]=os.environ[k]
    (root/'environment.sh').write_text(''.join('export '+k+'='+shlex.quote(v)+'\n'for k,v in env.items()))
    return {'context':str(root/'context.json'),'loaded_bundles':len(loaded),'compatibility_id':cid}


def flush(job, final=False):
    job=Path(job);root=job/'cache_continuity';context=read(root/'context.json');settings=read(context['settings_path']);store=Store(settings['store'])
    with file_lock(root/'flush.lock',timeout=3):
        cursor=read(root/'cursor.json')if(root/'cursor.json').exists()else{}
        export=ik.pack_delta(root/'exact',context['manifest'],root/'delta.json',cursor=cursor.get('ik'),max_entries=4096,max_bytes=128*1024*1024,deadline_seconds=60,source=context['owner'])
        chunks,offsets=overlay_chunks(context['overlay_shards'],cursor.get('overlay',{}),root/'chunks')
        row={'schema':SCHEMA,'owner':context['owner'],'compatibility_id':context['compatibility_id'],'at':time.time(),'final':final,
             'manifest':put(store,context['manifest']),'ik':put(store,root/'delta.json')if export['entry_count']else None,
             'ik_entries':export['entry_count'],'scan_complete':export['scan_complete'],'errors':export['failures'],
             'overlays':[{**{k:v for k,v in x.items()if k!='path'},'object':put(store,x['path'],'.jsonl')}for x in chunks],
             'base_overlay_files':{},'base_overlay_release':int(context['base_overlay_release']),'legacy_preserved':[]}
        # Copy only the pinned immutable overlay lineage; never modify originals.
        base=Path(context['base_overlay'])if context.get('base_overlay')else None
        if base and row['base_overlay_release']:
            from ispatialgym.sampling.v2 import map_tilt as M
            release_path=base/'releases'/('r'+str(row['base_overlay_release'])+'.json')
            row['base_overlay_files'][str(release_path.relative_to(base))]=put(store,release_path)
            for key,version in M.load_release(str(base),row['base_overlay_release']).items():
                path=base/key/('v'+str(version)+'.json');row['base_overlay_files'][str(path.relative_to(base))]=put(store,path)
        if final:
            # Also preserve old untyped caches outside the job. No import into the
            # compatible exchange is implied by this archival step.
            for label,directory in context['legacy_roots'].items():
                p=Path(directory)
                for f in([p]if p.is_file()else sorted(p.rglob('*'))):
                    if f.is_symlink()or not f.is_file()or f.suffix not in('.json','.npz'):continue
                    row['legacy_preserved'].append({'source':str(f),'kind':label,'object':put(store,f,f.suffix),'usable_shared':False})
        path=root/'upload.json';write(path,row);ref=put(store,path)
        store.put_verified(path,'pending/'+context['owner']+'/'+ref['sha256']+'.json')
        write(root/'cursor.json',{'ik':export['cursor'],'overlay':offsets})
        write(root/'uploaded.json',{'reference':ref,'row':row,'merged_published':False,'at':time.time()})
        return ref


def reconcile(store_root,durable):
    store=Store(store_root);durable=Path(durable);durable.mkdir(parents=True,exist_ok=True)
    completed=[]
    with file_lock(durable/'merge.lock',timeout=3):
        for key in store.list('pending'):
            ack=durable/'acks'/(hashlib.sha256(key.encode()).hexdigest()+'.json')
            if ack.exists():continue
            with tempfile.TemporaryDirectory(dir=durable)as tmp:
                path=Path(tmp)/'pending.json';store.get(key,path);row=read(path)
                if row['schema']!=SCHEMA:raise ValueError('unknown cache upload')
                owner=clean(row['owner']);cid=clean(row['compatibility_id']);objects=durable/'objects'
                def fetch(ref):return get(store,ref,objects/(ref['sha256']+Path(ref['key']).suffix))
                manifest_path=fetch(row['manifest']);manifest=compat.validate_manifest(read(manifest_path))
                if manifest['compatibility_sha256']!=cid:raise ValueError('incompatible upload')
                merged=durable/'merged'/cid;merged.mkdir(parents=True,exist_ok=True)
                imported=None
                if row['ik']:
                    imported=ik.ingest_bundle(fetch(row['ik']),durable/'exact',manifest,expected_sha256=row['ik']['sha256'],deadline_seconds=120,max_entries=4096,max_bytes=128*1024*1024)
                    if not imported['complete']:raise ValueError('cache merge incomplete')
                for x in row['legacy_preserved']:fetch(x['object'])
                overlay=merged/'overlay'
                # A new compatibility lineage starts from exactly its producer's
                # pinned baseline; later publications extend that lineage.
                if not(overlay/'releases').exists():
                    for name,ref in row['base_overlay_files'].items():
                        if Path(name).is_absolute()or'..'in Path(name).parts:raise ValueError('unsafe overlay baseline')
                        source=fetch(ref);target=overlay/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
                chunks=[str(fetch(x['object']))for x in row['overlays']]
                from ispatialgym.sampling.v2 import map_tilt as M
                if chunks:M.merge(chunks,str(overlay))
                overlay_release=M.latest_release(str(overlay));overlay_files={}
                if overlay_release:
                    rel=overlay/'releases'/('r'+str(overlay_release)+'.json');overlay_files[str(rel.relative_to(overlay))]=put(store,rel)
                    for mk,v in M.load_release(str(overlay),overlay_release).items():
                        p=overlay/mk/('v'+str(v)+'.json');overlay_files[str(p.relative_to(overlay))]=put(store,p)
                seed=merged/'seed.json';bundles=[];seed_cursor={};entries=0
                while True:
                    packed=ik.pack_delta(durable/'exact',manifest,seed,cursor=seed_cursor,max_entries=4096,max_bytes=128*1024*1024,deadline_seconds=120,source='durable_cluster_merge')
                    if packed['failures']:raise ValueError('published seed contains invalid entries')
                    if packed['entry_count']:bundles.append(put(store,seed));entries+=packed['entry_count']
                    if packed['scan_complete']:break
                    if packed['cursor']==seed_cursor:raise ValueError('published seed pagination stalled')
                    seed_cursor=packed['cursor']
                version={'schema':SCHEMA,'compatibility_id':cid,'ik_bundles':bundles,
                         'overlay_files':overlay_files,'overlay_release':overlay_release,'entries':entries,'at':time.time()}
                version['version']=hashlib.sha256(canonical(version)).hexdigest();vp=merged/'versions'/(version['version']+'.json');write(vp,version)
                store.put_verified(vp,'published/'+cid+'/'+version['version']+'.json')
                store.put_verified(vp,'published/'+cid+'/latest.json',immutable=False)
                receipt={'schema':SCHEMA,'owner':owner,'upload':{'key':key,'sha256':sha(path)},'durable_cluster':str(durable),
                         'version':version['version'],'merged_entries':imported['entries']if imported else 0,'overlay_release':overlay_release,
                         'final':row['final'],'checksums_verified':True,'at':time.time()}
                write(ack,receipt);store.put_verified(ack,'acks/'+owner+'/'+sha(path)+'.json');completed.append(receipt)
    return completed


def warm_prepare(settings_path,job):
    from scripts.campaign_20261110.infra_vast.worker import warm_cache as W
    settings=read(settings_path);warm=settings['warm'];job=Path(job);state=job/'cache_continuity/warm';state.mkdir(parents=True,exist_ok=True)
    code=Path(os.environ['ISGE_CODE']);b1k=(code/'.snapshot_b1k').read_text().strip()[:9]
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).splitlines()[0]
    key=re.sub(r'[^A-Za-z0-9.]','_',gpu)+'_'+b1k
    root=Path(warm['local_root']);kits=root/'kits';textures=root/'textures';kits.mkdir(parents=True,exist_ok=True);textures.mkdir(exist_ok=True)
    report=W.restore(Store(warm['store']),kits,textures,warm['abi'],key,state)
    baseline=[];read_kits=kits;read_textures=[]
    if not(kits/key/'.complete').exists():
        for old in warm.get('baseline_kit_roots',[]):
            if(Path(old)/key/'.complete').is_file():
                read_kits=Path(old);baseline.append({'kind':'kit','source':str(read_kits/key),'mode':'read_only_baseline'});break
    for old in warm.get('baseline_texture_roots',[]):
        if Path(old).is_dir():read_textures.append(old)
    read_textures.append(str(textures))
    context={'key':key,'kit_root':str(kits),'read_kit_root':str(read_kits),'textures':str(textures),'read_texture_roots':read_textures,
             'baseline':baseline,'settings':str(Path(settings_path).resolve()),'restore':report}
    write(state/'context.json',context)
    (state/'environment.sh').write_text('export KIT_TPL_NFS_ROOT='+shlex.quote(str(read_kits))+'\nexport KIT_TEX_STORE='+shlex.quote(read_textures[0])+
        '\nISG_WARM_EXTRA_TEXTURE_STORES=('+ ' '.join(shlex.quote(x)for x in read_textures[1:])+')\n')
    return report


def warm_flush(job,final=False):
    from scripts.campaign_20261110.infra_vast.worker import warm_cache as W
    job=Path(job);root=job/'cache_continuity';context=read(root/'context.json');warmctx=read(root/'warm/context.json')
    settings=read(context['settings_path']);warm=settings['warm'];store=Store(warm['store']);receipts=[]
    app=context.get('appdata')
    for name in([app]+[app+'.c'+str(i)for i in range(1,int(context.get('collectors',2))+1)]if app else[]):
        if not Path(name).exists():continue
        label=hashlib.sha256(name.encode()).hexdigest()[:16]
        captured=W.capture(name,warmctx['textures'],Path(name+'.cache_export')/label,final=final and (job/'sample.rc').exists()and(job/'sample.rc').read_text().strip()in('0','2'))
        if captured['kit']:
            receipt=W.package(store,captured['kit'],W.safe_kit_files(captured['kit']),kind='kit',key=warmctx['key'],abi=warm['abi'],state_dir=root/'warm',instance=settings['owner'])
            if receipt:receipts.append(receipt)
    report=W.export(store,warmctx['kit_root'],warmctx['textures'],warm['abi'],root/'warm',settings['owner'])
    report['receipts']+=receipts;report['final']=final
    write(root/'warm/latest.json',report)
    store.put_verified(root/'warm/latest.json','receipts/'+settings['owner']+'.json',immutable=False)
    return report


def warm_reconcile(store_root,durable):
    from scripts.campaign_20261110.infra_vast.worker import warm_cache as W
    store=Store(store_root);durable=Path(durable);durable.mkdir(parents=True,exist_ok=True);reports=[]
    with file_lock(durable/'warm_merge.lock',timeout=3):
        for key in store.list('manifests'):
            name=Path(key).name;receipt=durable/'receipts'/name
            if receipt.exists():continue
            with tempfile.TemporaryDirectory(dir=durable)as tmp:
                p=Path(tmp)/name;store.get(key,p)
                if sha(p)!=Path(key).stem:raise ValueError('warm manifest hash mismatch')
                m=read(p)
                if m['schema']!=W.SCHEMA:raise ValueError('foreign warm manifest')
                destination=durable/'versions'/Path(key).stem
                W.unpack_verified(store,m,destination)
                mp=destination/'manifest.json';shutil.copyfile(p,mp)
                # Only verified durable cluster artifacts enter the consumer index.
                published='published/'+key.removeprefix('manifests/')
                store.put_verified(p,published)
                r={'manifest':key,'manifest_sha256':sha(p),'archive_sha256':m['archive']['sha256'],'durable_cluster':str(destination),
                   'files':len(m['files']),'published':published,'checksums_verified':True,'at':time.time()}
                write(receipt,r);store.put_verified(receipt,'acks/'+Path(key).stem+'.json');reports.append(r)
    return reports


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','flush','watch','reconcile','warm-prepare','warm-flush','warm-reconcile']);p.add_argument('--job');p.add_argument('--settings',default=os.environ.get(ENV));p.add_argument('--store');p.add_argument('--durable');p.add_argument('--final',action='store_true');a=p.parse_args()
    if a.action=='warm-prepare':result=warm_prepare(a.settings,a.job)
    elif a.action=='warm-flush':result=warm_flush(a.job,a.final)
    elif a.action=='warm-reconcile':result=warm_reconcile(a.store,a.durable)
    elif a.action=='prepare':result=prepare(a.settings,a.job)
    elif a.action=='flush':result=flush(a.job,a.final)
    elif a.action=='reconcile':result=reconcile(a.store,a.durable)
    else:
        while not(Path(a.job)/'cache_continuity/drain').exists():
            try:
                flush(a.job);warm_flush(a.job)
            except Exception as exc:write(Path(a.job)/'cache_continuity/last_error.json',{'at':time.time(),'error':repr(exc)})
            for _ in range(90):
                if (Path(a.job)/'cache_continuity/drain').exists():break
                time.sleep(1)
        result={'cache':flush(a.job,True),'warm':warm_flush(a.job,True)}
    print(json.dumps(result))

if __name__=='__main__':main()
