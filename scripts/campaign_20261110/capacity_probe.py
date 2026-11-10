#!/usr/bin/env python3
"""Read-only cluster inventory; emits no credentials or unrelated user jobs."""
import collections, datetime, json, pathlib, re, subprocess

def output(args):
    return subprocess.check_output(args, text=True, timeout=30)
def kv(line):
    return dict(re.findall(r'(?:^|\s)(\w+)=(\S*)', line))
def tres(value):
    return dict(item.split('=', 1) for item in value.split(',') if '=' in item)
excluded={'svl9','viscam7','viscam11','viscam12','viscam14','viscam15','viscam16','viscam-hgx-1','viscam-hgx-2'}
allowed={'titanrtx','3090','a40','a5000','a6000','l40s'}
nodes=[]
for line in output(['scontrol','show','nodes','-o']).splitlines():
    row=kv(line)
    partitions=set(row.get('Partitions','').split(',')) & {'svl','viscam'}
    if not partitions: continue
    configured,allocated=tres(row.get('CfgTRES','')),tres(row.get('AllocTRES',''))
    model=next((k.split(':',1)[1] for k in configured if k.startswith('gres/gpu:')),None)
    free=max(0,int(configured.get('gres/gpu','0'))-int(allocated.get('gres/gpu','0')))
    cpus=max(0,int(row.get('CPUEfctv','0'))-int(row.get('CPUAlloc','0')))
    memory=max(0,int(row.get('RealMemory','0'))-int(row.get('AllocMem','0')))
    usable=model in allowed and row['NodeName'] not in excluded and not any(s in row.get('State','') for s in ['DRAIN','DOWN','FAIL'])
    slots=min(free,cpus//9,memory//(48*1024)) if usable else 0
    nodes.append(dict(node=row['NodeName'],partitions=sorted(partitions),model=model,state=row.get('State'),free_gpus=free,free_cpus=cpus,free_mem_mb=memory,eligible=usable,slots_9cpu_48gb=slots))
jobs=[]
for line in output(['squeue','-r','-u','minyeong,shiyuc','-h','-o','%u|%i|%j|%T|%P|%a|%b|%q']).splitlines():
    user,jid,name,state,part,acct,gres,qos=line.split('|')
    count=sum(int(x) for x in re.findall(r'gres/gpu(?::[^:,]+)?:(\d+)',gres))
    jobs.append(dict(user=user,id=jid,name=name,state=state,partition=part,account=acct,gpus=count,qos=qos))
summary=collections.defaultdict(lambda:collections.Counter())
for j in jobs:
    summary[j['user']][j['state']+'_gpus']+=j['gpus']
    summary[j['user']][j['state']+'_jobs']+=1
capacity=collections.Counter()
for n in nodes:
    if n['eligible']:
        for part in n['partitions']:
            capacity[part+'_free_gpus']+=n['free_gpus']
            capacity[part+'_slots_9cpu_48gb']+=n['slots_9cpu_48gb']
campaign_root=pathlib.Path('/vision/u/minyeong/projects/isg_home/campaign_20261110')
try:
    validation=json.loads((campaign_root/'validation/controller.json').read_text())
except (OSError,ValueError):
    validation={}
try:
    vast_campaign=json.loads((campaign_root/'vast/metrics.json').read_text())
except (OSError,ValueError):
    vast_campaign={}
scale_prebuild={}
for account,root in [('minyeong',campaign_root),('shiyuc',pathlib.Path('/vision/u/shiyuc/minyeong/campaign_20261110'))]:
    try:scale_prebuild[account]=json.loads((root/'scale_prebuild/controller.json').read_text())
    except (OSError,ValueError):pass
data_releases=[]
for path in sorted(campaign_root.glob('v2_release_*/release_manifest.json')):
    try:
        release=json.loads(path.read_text())
        reader=campaign_root/'validation'/(path.parent.name+'_reader.txt')
        if reader.exists() and 'ERRORS 0' in reader.read_text().splitlines():
            data_releases.append({k:v for k,v in release.items() if k!='files'} | {'path':str(path.parent),'reader_errors':0})
    except (OSError,ValueError):pass
print(json.dumps(dict(data_releases=data_releases,scale_prebuild=scale_prebuild,validation=validation,vast_campaign=vast_campaign,checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),summary=summary,capacity=capacity,nodes=nodes,jobs=jobs,qos=output(['sacctmgr','-nP','show','qos','where','name=svl,viscam','format=Name,MaxTRESPU,MaxJobsPU,MaxSubmitPU']),associations=output(['sacctmgr','-nP','show','assoc','where','user=minyeong,shiyuc','format=User,Account,Partition,QOS,DefaultQOS,GrpTRES,MaxTRES,MaxJobs,MaxSubmit'])),indent=2))
