"""Private transport for an unchanged materialized native geometry canary."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import uuid

SCHEMA='isg-vast-native-dependency-v1'
BASE=Path('/vision/u/minyeong/projects/isg_home/oracle_fix')
BASE_ENV={'ISPATIALGYM_GRASP_LIBRARY_V2':str(BASE/'v2maps_int2/glib'),
          'ISPATIALGYM_REACH_MAPS':str(BASE/'v2maps_int2/maps'),
          'ISPATIALGYM_REACH_MAP_ANCHORS':str(BASE/'v2maps_int2/anchors.json'),
          'ISPATIALGYM_MAP_OVERLAY':str(BASE/'map_overlay'),'ISPATIALGYM_MAP_OVERLAY_RELEASE':'3',
          'ISPATIALGYM_V2_ONLY':'1','ISPATIALGYM_REACH_MAPS_STRICT':'1','ISPATIALGYM_SUPPORT_POLICY':'grasp'}


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def library_environment(root,v2_root=None):
    environment=dict(BASE_ENV)
    if v2_root is not None:
        release=Path(v2_root).resolve()
        require(release.is_relative_to(Path(root).resolve()),'native library release outside campaign')
        environment.update(ISPATIALGYM_GRASP_LIBRARY_V2=str(release/'glib'),
                           ISPATIALGYM_REACH_MAPS=str(release/'maps'),
                           ISPATIALGYM_REACH_MAP_ANCHORS=str(release/'anchors.json'))
    return environment


def prepare(root,manifest,index,native,helper,original_wrapper,canonical_output,start_index,token,
            v2_root=None,v2_manifest_sha256=None):
    root,manifest,native,helper,wrapper=map(lambda p:Path(p).resolve(),(root,manifest,native,helper,original_wrapper))
    require(re.fullmatch(r'[A-Za-z0-9_-]+',token),'unsafe dependency token')
    require(all(p.is_relative_to(root) for p in (manifest,native,helper,wrapper)),'native dependency source outside campaign')
    require(native.is_relative_to(root/'code') and helper.is_relative_to(root/'code'),'native dependency requires frozen runtimes')
    entry=json.loads(manifest.read_text())['jobs'][index]
    require(not entry.get('blockers') and entry.get('production_released') is False,'native candidate is blocked or already production')
    argv=[item.replace('${MATERIALIZED}',str(manifest.parent)) for item in entry['sample_argv']]
    require(argv[:3]==['python','-m','ispatialgym.sampling.sample_isg'],'unsupported native prerequisite entry point')
    require('--no-write-back' in argv and '--out' in argv and '--customs' in argv and '--bddl-root' in argv,
            'native prerequisite lacks immutable explicit inputs')
    require('--start-index' not in argv and '--time-budget-s' not in argv,'native prerequisite already contains conflicting reservation')
    output=Path(canonical_output).absolute()
    require(output.is_relative_to(root/'materialized_runs') and output.parent.resolve().is_relative_to(root),
            'native output outside campaign materialization')
    require(not output.exists() and not output.is_symlink(),'native output already has evidence')
    original_output=argv[argv.index('--out')+1]
    require(original_output.startswith('${CAMPAIGN_OUT}/'),'native output does not use the owned materialization placeholder')
    suffix=Path(original_output.removeprefix('${CAMPAIGN_OUT}/')).parts
    require('..' not in suffix and tuple(output.parts[-len(suffix):])==suffix,'native output identity differs from source entry')
    job=root/'out/vast_native_dependency'/token;require(not job.exists(),'dependency token already exists');job.mkdir(parents=True)
    # The existing private worker streamer watches this name while native
    # evidence stays in the isolated result directory.
    (job/'sample.log').symlink_to('result/sample.log')
    roots=[]
    for flag in ('--customs','--bddl-root'):
        roots.append(Path(argv[argv.index(flag)+1]))
    environment=dict(entry.get('environment') or {})
    require(set(environment)<={'ISPATIALGYM_STABLE_SCENE_DIR','ISPATIALGYM_INTERIORS'},
            'unrecognized native candidate environment')
    if environment.get('ISPATIALGYM_INTERIORS'):
        registry=Path(environment['ISPATIALGYM_INTERIORS'])
        require(registry.suffix=='.json' and registry.is_file(),
                'native interior registry must be an explicit JSON file')
    roots += [Path(value) for value in environment.values()]
    paths=[manifest,wrapper,native/'.snapshot_commit',helper/'scripts/campaign_20261110/vast_native_dependency.py',
           helper/'scripts/campaign_20261110/job_native_dependency.sh']
    for path in roots:
        require(path.resolve().is_relative_to(root) and path.exists(),'native source dependency outside campaign or missing')
        paths += [path] if path.is_file() else [p for p in path.rglob('*') if p.is_file()]
    files={};pending=list(paths)
    def referenced_files(value):
        if isinstance(value,dict):
            for item in value.values():yield from referenced_files(item)
        elif isinstance(value,list):
            for item in value:yield from referenced_files(item)
        elif isinstance(value,str) and value.startswith(str(root)+'/'):
            path=Path(value)
            if (path.suffix in ('.json','.png','.npz','.npy','.bddl','.py','.sh') and path.is_file()
                    and not path.is_relative_to(root/'vast')):
                yield path
    while pending:
        path=pending.pop()
        if str(path) in files:continue
        require(len(files)<10000,'native dependency closure exceeds bound')
        require(path.resolve().is_relative_to(root),'native input symlink escapes campaign')
        files[str(path)]=sha(path)
        if path.is_symlink():files[str(path.resolve())]=sha(path)
        if path.suffix=='.json':pending.extend(referenced_files(json.loads(path.read_text())))
    argv[argv.index('--out')+1]=str(job/'result')
    argv += ['--start-index',str(start_index),'--time-budget-s','1800']
    library_files={}
    if v2_root is not None:
        release=Path(v2_root).resolve();release_manifest=release/'release_manifest.json'
        require(release.is_relative_to(root) and v2_manifest_sha256 and sha(release_manifest)==v2_manifest_sha256,
                'native library release manifest identity differs')
        library_files[str(release_manifest)]=v2_manifest_sha256
        for relative,expected in json.loads(release_manifest.read_text())['files'].items():
            path=release/relative
            require(path.resolve().is_relative_to(release) and path.is_file() and sha(path)==expected,
                    'native library release file differs: '+relative)
            library_files[str(path)]=expected
        require(str(release/'anchors.json') in library_files,'native library release lacks pinned anchors')
    environment.update(library_environment(root,v2_root))
    environment.update(ISG_VEC_ASSET_CACHE_DIR=str(root/'asset_cache'),ISG_VEC_ASSET_CACHE_MODE='rw',
                       ISPATIALGYM_SAMPLING_CACHE=str(job/'sampling_cache'))
    env={'ISGE_CODE':str(native),'ISGF_ROOT':str(root),'ISGF_NATIVE_HELPER':str(helper)}
    (job/'job.env').write_text(''.join(k+'='+shlex.quote(v)+'\n' for k,v in env.items()))
    files[str(job/'job.env')]=sha(job/'job.env')
    value={'schema':SCHEMA,'campaign_id':'campaign_20261110','campaign_root':str(root),'job_dir':str(job),
           'snapshot':str(native),'helper_snapshot':str(helper),'source_manifest':str(manifest),'entry_index':index,
           'original_wrapper':str(wrapper),'original_entry':entry,'canonical_output':str(output),'argv':argv,
           'fresh_start_index':start_index,'environment':environment,'files':files,'scenes':[entry['scene']],
           'v2_release_root':str(Path(v2_root).resolve()) if v2_root else None,'library_files':library_files,
           'shared_anchor_sha256':sha(environment['ISPATIALGYM_REACH_MAP_ANCHORS']),
           'prepared_at_unix':time.time(),'runtime_adjustments':['image_native_apps','private_warm_kit_cache','worker_cpu_affinity','owned_output_relocation']}
    write(job/'native_inputs.json',value)
    return job


def verify(job,root=None):
    job=Path(job).resolve();value=json.loads((job/'native_inputs.json').read_text());owned=Path(root or value['campaign_root']).resolve()
    require(value['schema']==SCHEMA and value['campaign_root']==str(owned),'native dependency campaign identity differs')
    require(job.is_relative_to(owned/'out/vast_native_dependency') and value['job_dir']==str(job),'native dependency job ownership differs')
    for name,expected in value['files'].items():
        path=Path(name);require(path.resolve().is_relative_to(owned) and path.is_file() and sha(path)==expected,'native dependency input bytes changed: '+name)
    expected_environment=library_environment(owned,value.get('v2_release_root'))
    require(all(value['environment'].get(key)==expected for key,expected in expected_environment.items()),'native shared library identity differs')
    for name,expected in value.get('library_files',{}).items():
        path=Path(name);require(path.resolve().is_relative_to(owned) and path.is_file() and sha(path)==expected,
                                'native library bytes changed: '+name)
    require(sha(value['environment']['ISPATIALGYM_REACH_MAP_ANCHORS'])==value['shared_anchor_sha256'],'native shared anchor bytes changed')
    return value


def enqueue(job,root,gpu='RTX 4090'):
    root,job=Path(root).resolve(),Path(job).resolve();value=verify(job,root);receipt=job/'vast_submission.json'
    if receipt.exists():
        old=json.loads(receipt.read_text());require(sha(root/'vast/queue'/(str(old['id'])+'.json'))==old['record_sha256'],'native queue identity changed');return str(old['id'])
    ids=root/'vast/queue/ids';ids.mkdir(parents=True,exist_ok=True);jid=max([900000000]+[int(p.name) for p in ids.iterdir() if p.name.isdigit()])+1
    while True:
        try:(ids/str(jid)).mkdir();break
        except FileExistsError:jid+=1
    record={'id':jid,'job_kind':'native_dependency','name':'isg261110_native_prereq',
            'script':str(Path(value['helper_snapshot'])/'scripts/campaign_20261110/job_native_dependency.sh'),
            'job_dir':str(job),'time_s':3000,'cpus':9,'mem':'48G','acct':'vision','part':'svl',
            'log':str(root/'slurm'/('isg261110_native_prereq_'+str(jid)+'.log')),'cwd':str(root),
            'submitted':time.time(),'submitted_by':'campaign_native_dependency','gpu':gpu,
            'native_inputs_sha256':sha(job/'native_inputs.json')}
    target=root/'vast/queue'/(str(jid)+'.json');write(target,record);write(receipt,{'id':jid,'record_sha256':sha(target),'activated':False})
    return str(jid)


def bind_output(job,root):
    value=verify(job,root);output=Path(value['canonical_output']);target=Path(job).resolve()/'result'
    require(output.parent.resolve().is_relative_to(Path(root).resolve()),'native output parent escaped campaign')
    if output.is_symlink():require(output.resolve()==target,'native output redirected elsewhere');return
    require(not output.exists(),'native output has prior evidence');output.parent.mkdir(parents=True,exist_ok=True);output.symlink_to(target)


def run(job):
    value=verify(job);job=Path(job).resolve();output=job/'result';output.mkdir(parents=True,exist_ok=False)
    os.environ.update(value['environment']);Path(value['environment']['ISPATIALGYM_SAMPLING_CACHE']).mkdir(parents=True,exist_ok=True)
    argv=list(value['argv']);argv[0]=sys.executable
    write(output/'command.json',{'argv':argv,'manifest_sha256':value['files'][value['source_manifest']],
                               'snapshot':value['snapshot'],'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
                               'source_transport':str(job/'native_inputs.json'),
                               'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    with (output/'sample.log').open('w') as log:
        try:rc=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=2400).returncode
        except subprocess.TimeoutExpired:rc=124
    tros=list(output.rglob('*-tro_state.json'))
    report={'returncode':rc,'tro_count':len(tros),'tros':[str(p) for p in tros],
            'class_id':value['original_entry']['class_id'],'fresh_start_index':value['fresh_start_index'],
            'validation_only':True,'finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write(output/'sampling_gate.json',report);print(json.dumps(report));return 0 if rc==0 and len(tros)==1 else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify',type=Path);parser.add_argument('--run',type=Path);args=parser.parse_args()
    if args.verify:verify(args.verify);print('immutable native dependency inputs verified')
    elif args.run:sys.exit(run(args.run))
    else:parser.error('choose --verify or --run')
