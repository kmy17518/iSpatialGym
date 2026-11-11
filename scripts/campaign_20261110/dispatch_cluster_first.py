"""Fresh two-account availability before a new direct Vast dispatch claim.

Existing allocations and pending-job migration retain their exclusive owners.
This gate only arbitrates work that has never been submitted to any backend.
"""
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

CLUSTER_BACKENDS = ('minyeong_svl', 'shiyuc_svl', 'minyeong_viscam', 'shiyuc_viscam')
COMPATIBLE_MODELS = {'titanrtx', '3090', 'a40', 'a5000', 'a6000', 'l40s'}


def option(command, flag, default):
    return command[command.index(flag)+1] if flag in command else default


def assigned_literal(path, name):
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('missing static resource constraint: '+name)


def requirements(payload):
    command=payload['command'];code=Path(payload['identity']['code'])
    cpus=int(option(command,'--cpus',max(8,3*(int(option(command,'--kcollect',1))+1))))
    memory=str(option(command,'--mem','48G')).upper()
    match=re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)([MG]?)',memory)
    if not match or cpus<=0:raise ValueError('unrecognized exact CPU/memory requirement')
    memory_mb=float(match[1])*(1024 if match[2]=='G' else 1)
    if memory_mb<=0:raise ValueError('unbounded memory request needs explicit compatibility evidence')
    native=code/'scripts/trainfast/submit_fast.py'
    wrapper=Path(__file__).parent/'infra_dispatch_bin/sbatch'
    excluded=set(assigned_literal(native,'EXCLUDE').split(','))|set(assigned_literal(wrapper,'excluded'))
    return {'cpus':cpus,'memory_mb':memory_mb,'excluded_nodes':sorted(excluded),
            'native_submit_sha256':hashlib.sha256(native.read_bytes()).hexdigest(),
            'scheduler_wrapper_sha256':hashlib.sha256(wrapper.read_bytes()).hexdigest()}


def probe():
    queries={'jobs':['squeue','-r','-u','minyeong,shiyuc','-h','-o','%u|%P|%T|%b|%q|%R|%i'],
             'nodes':['scontrol','show','nodes','-o']}
    started=time.time()
    def run(argv):
        return subprocess.run(argv,capture_output=True,text=True,timeout=40,check=True).stdout
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures={k:pool.submit(run,v)for k,v in queries.items()}
            raw={k:f.result()for k,f in futures.items()}
        jobs=[line.split('|')for line in raw['jobs'].splitlines()if line.strip()]
        if any(len(row)!=7 for row in jobs):raise ValueError('scheduler row schema changed')
        nodes=[dict(re.findall(r'(?:^|\s)(\w+)=(\S*)',line))for line in raw['nodes'].splitlines()if line.strip()]
        if not nodes:raise ValueError('no node observations')
        return {'started_at':started,'observed_at':started,'finished_at':time.time(),'state':'available','jobs':jobs,'nodes':nodes,
                'query_sha256':{k:hashlib.sha256(v.encode()).hexdigest()for k,v in raw.items()}}
    except (OSError,ValueError,subprocess.SubprocessError) as error:
        return {'started_at':started,'observed_at':time.time(),'state':'unavailable','error':str(error),
                'jobs':[],'nodes':[]}


def gpu_count(value):
    return sum(int(x)for x in re.findall(r'gres/gpu(?::[^:,]+)?:(\d+)',value))


def slots(node, partition, demand):
    if partition not in node.get('Partitions','').split(',') or node.get('NodeName') in demand['excluded_nodes']:
        return 0
    state=node.get('State','')
    if not any(s in state for s in ('IDLE','MIXED','ALLOCATED')) or any(s in state for s in ('DRAIN','DOWN','FAIL','POWER','RESERVED','PLANNED','MAINT','NOT_RESPOND')):
        return 0
    cfg=dict(x.split('=',1)for x in node.get('CfgTRES','').split(',')if '='in x)
    allocated=dict(x.split('=',1)for x in node.get('AllocTRES','').split(',')if '='in x)
    models={k.split(':',1)[1]for k in cfg if k.startswith('gres/gpu:')}
    if not models or not models<=COMPATIBLE_MODELS or partition=='svl' and models!={'titanrtx'}:return 0
    free_gpu=max(0,int(cfg.get('gres/gpu','0'))-int(allocated.get('gres/gpu','0')))
    free_cpu=max(0,int(node.get('CPUEfctv',node.get('CPUTot','0')))-int(node.get('CPUAlloc','0')))
    free_mem=max(0,int(node.get('RealMemory','0'))-int(node.get('AllocMem','0')))
    return min(free_gpu,free_cpu//demand['cpus'],int(free_mem//demand['memory_mb']))


def assess(payload, observation, workers, demand, *, now=None):
    from scripts.campaign_20261110.infra_dispatch import backend_capacity
    now=time.time()if now is None else now
    age=now-observation['observed_at']
    base={'observed_at':now,'observation_age_seconds':age,'demand':demand,
          'cluster_probe':{k:v for k,v in observation.items()if k not in ('jobs','nodes')},
          'owner':'/root/compute','minimum_pending_seconds':0,
          'scope':'New direct Vast claim only; existing and ambiguous allocations are not resubmitted.'}
    if observation['state']!='available' or not 0<=age<=60:
        return {**base,'allow_vast':True,'preferred_cluster_backends':[],
                'reason':'cluster_availability_unavailable','accounts_checked':['minyeong','shiyuc'],
                'next_action':'Repair fresh cluster observation; independent funded work may continue on Vast.'}
    rows=[]
    for backend in CLUSTER_BACKENDS:
        user,partition=backend.split('_');w=workers.get(backend,{})
        worker_age=now-w.get('observed_at',0)
        quota=backend_capacity(backend,observation['jobs'])
        candidates=[{'node':n['NodeName'],'slots':slots(n,partition,demand)}for n in observation['nodes']]
        inventory=sum(n['slots']for n in candidates)
        pending=[r for r in observation['jobs']if r[0]==user and partition in r[1].split(',')
                 and r[2]=='PENDING' and gpu_count(r[3])>0 and
                 (r[5] in ('(Priority)','(Resources)','Priority','Resources') or r[5].lstrip('(').startswith(('QOSMax','AssocMax')))]
        if backend not in payload['backends']:reason='intent_cluster_transport_not_enabled'
        elif not 0<=worker_age<=120:reason='cluster_dispatcher_unobserved'
        elif w.get('phase')=='submitting' and worker_age>20:reason='cluster_dispatcher_busy'
        elif quota<=0:reason='account_partition_limit'
        elif pending:reason='independent_gpu_work_already_pending'
        elif inventory<=0:reason='no_compatible_unallocated_inventory'
        else:reason='available'
        rows.append({'backend':backend,'reason':reason,'quota_headroom':quota,
                     'inventory_slots':inventory,'inventory_is_reservation':False,
                     'worker_observation':w,'worker_age_seconds':worker_age,
                     'pending_job_ids':[r[6]for r in pending],
                     'compatible_nodes':[r for r in candidates if r['slots']]})
    preferred=[r['backend']for r in rows if r['reason']=='available']
    return {**base,'allow_vast':not preferred,'preferred_cluster_backends':preferred,
            'accounts_checked':['minyeong','shiyuc'],'backends':rows,
            'reason':'compatible_cluster_first'if preferred else'no_immediate_compatible_cluster_route',
            'next_action':'Cluster owner claims this unchanged intent.'if preferred else
            'Use funded Vast for this independent intent; repair any missing cluster transport or stale dispatcher separately.'}


def check(root,payload,observation=None):
    root=Path(root)
    if observation is None or not 0<=time.time()-observation.get('observed_at',0)<=60:
        observation=probe()
    workers={}
    for name in CLUSTER_BACKENDS:
        try:workers[name]=json.loads((root/'dispatch/backend_observations'/(name+'.json')).read_text())
        except (OSError,ValueError):pass
    try:
        demand=requirements(payload)
    except (OSError,ValueError,TypeError,KeyError) as error:
        return {'observed_at':time.time(),'allow_vast':False,'preferred_cluster_backends':[],
                'reason':'unverified_resource_requirements','error':str(error),
                'owner':'/root/compute','accounts_checked':['minyeong','shiyuc'],
                'cluster_probe':{k:v for k,v in observation.items()if k not in ('jobs','nodes')},
                'next_action':'Resolve the exact native resource/exclusion declaration before direct Vast dispatch.'}
    return assess(payload,observation,workers,demand)
