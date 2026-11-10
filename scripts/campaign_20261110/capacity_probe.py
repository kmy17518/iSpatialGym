#!/usr/bin/env python3
"""Read-only cluster inventory; emits no credentials or unrelated user jobs."""
import collections, datetime, json, pathlib, re, shlex, subprocess

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
        if (path.parent/'release_block.json').exists():continue
        release=json.loads(path.read_text())
        reader=campaign_root/'validation'/(path.parent.name+'_reader.txt')
        if reader.exists() and 'ERRORS 0' in reader.read_text().splitlines():
            data_releases.append({k:v for k,v in release.items() if k!='files'} | {'path':str(path.parent),'reader_errors':0})
    except (OSError,ValueError):pass
acceptance={'counts':{},'templates':0,'sources':[],'errors':[]}
counts=collections.Counter()
template_keys=set()
group_path=campaign_root/'dataset/group_counts.json'
if group_path.exists():
    try:
        committed=json.loads(group_path.read_text())
        assert committed['campaign_id']=='campaign_20261110'
        counts.update(committed['counts'])
        exports=[json.loads(p.read_text()) for p in (campaign_root/'dataset/groups').glob('*.json')]
        members=[m for group in exports for m in group['members']]
        assert len(exports)==committed['groups'] and len(members)==sum(committed['counts'].values()), 'group export/count mismatch'
        template_keys.update(m['validation_key']['template_sha256'] for m in members)
        acceptance['sources'].append({'path':str(group_path),'mtime':group_path.stat().st_mtime})
    except (OSError,ValueError,KeyError,AssertionError) as exc:acceptance['errors'].append(str(exc))
camera_ids=set()
for camera_path in [campaign_root/'dataset/camera_quota.json',campaign_root/'dataset/camera_obstructed_quota.json']:
    if not camera_path.exists():continue
    try:
        state=json.loads(camera_path.read_text())
        assert state['campaign_id']=='campaign_20261110'
        for iid,member in state['accepted'].items():
            assert iid not in camera_ids, 'camera instance appears in two quota states'
            camera_ids.add(iid)
            counts[member['split']]+=1
            template_keys.add(member['validation_key']['template_sha256'])
        acceptance['sources'].append({'path':str(camera_path),'mtime':camera_path.stat().st_mtime})
    except (OSError,ValueError,KeyError,AssertionError) as exc:acceptance['errors'].append(str(exc))
acceptance.update(counts=dict(counts),templates=len(template_keys))
coverage={}
coverage_errors=[]
for gate in validation.get('gates',{}).values():
    job=pathlib.Path(gate.get('source_job_dir',''))
    if not job.is_relative_to(campaign_root/'out'):continue
    try:
        lines=(job/'job.env').read_text().splitlines()
        bindings=[shlex.split(line.split('=',1)[1]) for line in lines if line.startswith('ISPATIALGYM_CAMPAIGN_CONFIG=')]
        if len(bindings)!=1 or len(bindings[0])!=1:continue
        config=json.loads(pathlib.Path(bindings[0][0]).read_text())
        template=config['template'];key=config['validation_key']['template_sha256']
        entry=coverage.setdefault(key,{'class_id':template['class_id'],'task':template['task'],'split':template['split'],
                                      'demo_passed':False,'replay_passed':False,'accepted':key in template_keys})
        entry['demo_passed'] |= gate.get('demo_successes',0)>0
        entry['replay_passed'] |= gate.get('status')=='validated' and gate.get('gate',{}).get('success') is True
    except (OSError,ValueError,KeyError) as exc:
        coverage_errors.append({'job':str(job),'error':str(exc)[:180]})
annotated_coverage={'templates':coverage,'errors':coverage_errors,
                    'demo_passed':sum(x['demo_passed'] for x in coverage.values()),
                    'replay_passed':sum(x['replay_passed'] for x in coverage.values()),
                    'accepted':len(template_keys)}
family_controller={}
try:
    raw=json.loads((campaign_root/'controllers/families/state.json').read_text())
    family_controller={'updated_at_unix':raw['updated_at_unix'],'probe_error':raw.get('probe_error'),'families':{key:{'status':value.get('status'),'error':value.get('error'),'completed_cycles':len(value.get('history',[])), 'active_job':(value.get('current',{}).get('group') or value.get('current',{}).get('anchor') or {}).get('allocation',{}).get('jobid')} for key,value in raw['families'].items()}}
except (OSError,ValueError,KeyError):pass
try:throughput=json.loads((campaign_root/'infra/throughput.json').read_text())
except (OSError,ValueError):throughput={}
print(json.dumps(dict(annotated_coverage=annotated_coverage,acceptance=acceptance,family_controller=family_controller,throughput=throughput,data_releases=data_releases,scale_prebuild=scale_prebuild,validation=validation,vast_campaign=vast_campaign,checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),summary=summary,capacity=capacity,nodes=nodes,jobs=jobs,qos=output(['sacctmgr','-nP','show','qos','where','name=svl,viscam','format=Name,MaxTRESPU,MaxJobsPU,MaxSubmitPU']),associations=output(['sacctmgr','-nP','show','assoc','where','user=minyeong,shiyuc','format=User,Account,Partition,QOS,DefaultQOS,GrpTRES,MaxTRES,MaxJobs,MaxSubmit'])),indent=2))
