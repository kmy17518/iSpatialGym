#!/usr/bin/env python3
"""Stage only the campaign's new Vast prefix, reusing frozen static URLs read-only."""
import concurrent.futures,datetime,hashlib,io,json,os,pathlib,tarfile
import boto3
from urllib.parse import urlparse
ROOT=pathlib.Path('/vision/u/minyeong/projects/isg_home/campaign_20261110')
VJ=ROOT/'vast'
TOOL=pathlib.Path(os.environ['VTOOL']).resolve()
assert str(TOOL).startswith(str(ROOT/'code')+'/'), 'Use frozen campaign-owned tooling'
assert os.environ['JOB_ID']=='og-isg261110'
job=urlparse(os.environ['JOB_S3'])
assert job.path=='/jobs/og-isg261110'
client=boto3.client('s3',endpoint_url=os.environ['S3_ENDPOINT_URL'],region_name='auto')
for name in ('queue','state/jobs','ledger','logs','market','bundle','results'):(VJ/name).mkdir(parents=True,exist_ok=True)
def put(key,body):
    client.put_object(Bucket=job.netloc,Key=job.path.strip('/')+'/'+key,Body=body.encode() if isinstance(body,str) else body)
source=pathlib.Path('/vision/u/minyeong/projects/isg_home/oracle_fix/vast/bundle')
phase1=(source/'boot_list.txt').read_text()
phase2=''.join(line+'\n' for line in (source/'boot_list2.txt').read_text().splitlines() if '/cache/' not in line)
assert all('/cache/' not in line for line in phase1.splitlines()), 'Do not seed prior sampling caches'
urls=[line.split()[0] for line in (phase1+phase2).splitlines() if line.strip()]
def inspect(url):
    parsed=urlparse(url);head=client.head_object(Bucket=parsed.netloc,Key=parsed.path.lstrip('/'))
    return {'url':url,'etag':head['ETag'],'bytes':head['ContentLength'],'last_modified':head['LastModified'].isoformat()}
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:provenance=list(pool.map(inspect,urls))
put('bundle/boot_list.txt',phase1);put('bundle/boot_list1b.txt','');put('bundle/boot_list2.txt',phase2)
client.copy_object(Bucket=job.netloc,Key=job.path.strip('/')+'/bundle/tools/rsync.tar.zst',CopySource={'Bucket':'isg-demos','Key':'jobs/og-isgof-vast-20261108/bundle/tools/rsync.tar.zst'})
blob=io.BytesIO()
with tarfile.open(fileobj=blob,mode='w:gz') as archive:
    for path in sorted((TOOL/'worker').iterdir()):
        if path.is_file() and path.suffix not in ('.pyc',):archive.add(path,arcname=path.name)
worker=blob.getvalue();put('bundle/worker.tar.gz',worker)
manifest={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tool':str(TOOL),'worker_sha256':hashlib.sha256(worker).hexdigest(),'static_sources':provenance,'sampling_cache_seeded':False}
(VJ/'bundle/provenance.json').write_text(json.dumps(manifest,indent=2)+'\n');put('bundle/provenance.json',json.dumps(manifest))
config={'gpus':['RTX 4090','RTX 5090'],'default_gpu':'RTX 4090','max_instances':2,'budget_usd':3000,'transfer_reserve_usd':300,'max_dph_gpu':1.5,'disk_gb':280,'idle_destroy_s':420,'boot_max_s':3300,'launch_per_tick':2,'min_inet_down':700,'vram_per_job_gb':20,'ram_per_job_gb':40,'cpus_per_job':9,'max_jobs_per_instance':1,'max_worker_lifetime_s':21600,'retries':2,'search_every_s':600}
configpath=VJ/'config.json'
if not configpath.exists():configpath.write_text(json.dumps(config,indent=2)+'\n')
put('bundle/boot_ready',datetime.datetime.now(datetime.timezone.utc).isoformat()+'\n')
link=VJ/'tool'
if link.is_symlink():
    assert link.resolve()==TOOL,'Existing campaign tool differs; explicitly freeze a new version before changing it'
else:link.symlink_to(TOOL)
print(json.dumps({'ready':True,'tool':str(TOOL),'static_archives':len(provenance),'static_download_bytes':sum(p['bytes'] for p in provenance),'new_prefix':os.environ['JOB_S3'],'worker_sha256':manifest['worker_sha256']}))
