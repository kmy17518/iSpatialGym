"""Typed transport for exact, already-positive contact-library map units.

Native ContactMap configuration and pose identities are preserved. Only the
library's object.source path is relocated; no library training weights are sent.
Queue activation requires exclusive cluster-owner handoff evidence.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.campaign_20261110.vast_prebuild import require, sha, write

SCHEMA = 'isg-vast-contact-map-prebuild-v1'
DENIED = {'.pt', '.pth', '.ckpt', '.safetensors'}


def library_environment(root, v2_root=None):
    return {}


def validate_unit(doc, index, native):
    require(doc.get('campaign_id') == 'campaign_20261110' and doc.get('production_authorization') is False,
            'contact request campaign/purpose differs')
    require(doc['native_runtime'] == str(native), 'contact native runtime differs')
    require(type(index) is int and 0 <= index < len(doc['units']), 'invalid contact unit')
    unit = doc['units'][index]
    require(unit['id'] == index and unit['native_runtime'] == str(native), 'contact unit runtime/index differs')
    require(unit['state'] == 'ready_exact_map_dependency' and unit['positive_entries'] > 0,
            'contact library is not ready and positive')
    require(unit['production_authorization'] is False and unit['configuration_validated'] is False,
            'dependency is not configuration validation')
    require(unit['arm'] in ('left', 'right') and unit['expected_map_config']['arm'] == unit['arm'],
            'contact arm identity differs')
    require(float(unit['height_key']) == unit['height'] and unit['pose_set']['digest'], 'height/pose identity differs')
    require(re.fullmatch(r'[A-Za-z0-9_.-]+', unit['entry_id']), 'unsafe contact entry')
    require(unit['map_kind'] in ('push_blade', 'touch_push', 'axial_brace', 'touch_brace', 'floor_push', 'wall_brace'),
            'unsupported exact contact-map kind')
    require(unit['library_kind'] in ('push', 'brace'), 'unsupported contact library kind')
    require(unit['expected_map_config']['seeds'] == 8,
            'contact-map sampling grid differs')
    fixed = {'push_blade': ('push', 'right'), 'touch_push': ('push', 'right'),
             'floor_push': ('push', 'right'), 'axial_brace': ('brace', 'left'), 'touch_brace': ('brace', 'left')}
    require(unit['map_kind'] not in fixed or (unit['library_kind'], unit['arm']) == fixed[unit['map_kind']],
            'contact kind/library/arm differs')
    return unit


def prepare(root, request, index, native, helper, token):
    root, request, native, helper = [Path(p).resolve() for p in (root, request, native, helper)]
    require(root.name == 'campaign_20261110' and request.is_relative_to(root/'training_preparation'), 'wrong contact campaign')
    require(native.is_relative_to(root/'code') and helper.is_relative_to(root/'code'), 'unfrozen contact runtime')
    require(re.fullmatch(r'[A-Za-z0-9_-]+', token), 'unsafe contact token')
    doc = json.loads(request.read_text()); unit = validate_unit(doc, index, native)
    files = dict(unit['source_files'])
    for ref in (unit['source_library'], unit['source_physics_audit'], unit['selected_manifest']):
        files[ref['path']] = ref['sha256']
    for name, expected in files.items():
        p = Path(name)
        require(p.resolve().is_relative_to(root) and p.is_file() and p.suffix.lower() not in DENIED,
                'contact input escaped or includes a weight payload')
        require(sha(p) == expected, 'contact source changed: '+name)
    original = json.loads(Path(unit['source_library']['path']).read_text())
    require(original['object'] == unit['object'], 'contact library object identity differs')
    source_model = Path(unit['model_dir']).resolve()
    require(original['object']['source'] == str(source_model), 'contact model source differs')
    job = root/'out/vast_contact_map'/token
    require(not job.exists(), 'contact token already has evidence')
    job.mkdir(parents=True)
    model = job/'inputs/glib/models'/unit['entry_id']; model.mkdir(parents=True)
    model_files = {}
    for name, expected in files.items():
        src = Path(name)
        if src.parent == source_model:
            dst = model/src.name; dst.write_bytes(src.read_bytes()); require(sha(dst) == expected, 'model copy changed')
            model_files[str(dst)] = expected
    require(all((model/n).is_file() for n in ('library.json', 'geometry.json', 'geometry.obj', 'rests.json')),
            'incomplete exact model geometry closure')
    relocated = copy.deepcopy(original); relocated['object']['source'] = str(model)
    check = copy.deepcopy(relocated); check['object']['source'] = original['object']['source']
    require(check == original, 'contact library changed beyond source relocation')
    library = job/'inputs/contacts'/unit['library_kind']/'models'/unit['entry_id']/'library.json'
    write(library, relocated); model_files[str(library)] = sha(library)
    env = {'ISGE_CODE':str(native), 'ISGF_ROOT':str(root), 'ISGF_NATIVE_HELPER':str(helper)}
    (job/'job.env').write_text(''.join(k+'='+shlex.quote(v)+'\n' for k,v in env.items()))
    for p in (request, native/'.snapshot_commit', helper/'scripts/campaign_20261110/vast_contact_map.py',
              helper/'scripts/campaign_20261110/vast_prebuild.py', helper/'scripts/campaign_20261110/job_contact_map.sh', job/'job.env'):
        files[str(p)] = sha(p)
    value = {'schema':SCHEMA, 'campaign_id':root.name, 'campaign_root':str(root), 'job_dir':str(job),
             'snapshot':str(native), 'helper_snapshot':str(helper), 'source_plan':str(request), 'unit_index':index,
             'unit':unit, 'files':files, 'library_files':model_files, 'private_contact_root':str(job/'inputs/contacts'),
             'native_library':str(library), 'source_relocation':{'before':str(source_model),'after':str(model)},
             'kind':'contact_maps', 'scenes':[], 'time_limit_s':6900, 'new_native_instances':0,
             'benchmark_credit':0, 'production_released':False}
    write(job/'native_inputs.json', value); (job/'sample.log').symlink_to('result/native.log')
    return job


def verify(job, root=None):
    job = Path(job).resolve(); value = json.loads((job/'native_inputs.json').read_text())
    root = Path(root or value['campaign_root']).resolve()
    require(value['schema'] == SCHEMA and value['campaign_id'] == root.name == 'campaign_20261110', 'contact campaign differs')
    require(value['campaign_root'] == str(root) and job.is_relative_to(root/'out/vast_contact_map') and value['job_dir'] == str(job),
            'contact output ownership differs')
    require(value['new_native_instances'] == 0 and value['benchmark_credit'] == 0 and value['production_released'] is False,
            'contact purpose differs')
    for name, expected in {**value['files'], **value['library_files']}.items():
        p = Path(name)
        require(p.resolve().is_relative_to(root) and p.suffix.lower() not in DENIED and sha(p) == expected,
                'contact input changed: '+name)
    doc = json.loads(Path(value['source_plan']).read_text())
    require(value['unit'] == validate_unit(doc, value['unit_index'], Path(value['snapshot'])), 'contact source unit changed')
    original = json.loads(Path(value['unit']['source_library']['path']).read_text())
    relocated = json.loads(Path(value['native_library']).read_text())
    require(relocated['object']['source'] == value['source_relocation']['after'], 'contact relocation changed')
    relocated['object']['source'] = value['source_relocation']['before']
    require(relocated == original, 'contact library semantic identity differs')
    return value


def validate_handoff(value, handoff, root):
    """An independent transport cannot create its own cluster exclusion proof."""
    root = Path(root); p = Path(handoff).resolve()
    require(p.is_relative_to(root/'infra/capacity_recovery/contact_map_handoffs'), 'contact handoff outside owned registry')
    proof = json.loads(p.read_text())
    require(proof.get('schema') == 'isg-contact-map-exclusive-handoff-v1' and proof.get('campaign_id') == root.name,
            'contact handoff schema differs')
    require(proof['unit_index'] == value['unit_index'] and proof['source_plan'] ==
            {'path':value['source_plan'],'sha256':sha(value['source_plan'])}, 'handoff exact request differs')
    require(proof.get('cluster_execution_excluded') is True and proof.get('owner') == '/root/templates_assets',
            'contact cluster owner has not excluded this unit')
    require(proof.get('mode') in ('never_submitted_retired','never_started_cancelled'), 'ambiguous prior execution')
    require(proof.get('evidence'), 'contact owner evidence missing')
    for ref in proof['evidence']:
        require(Path(ref['path']).resolve().is_relative_to(root) and sha(ref['path']) == ref['sha256'], 'handoff evidence changed')
    if proof['mode'] == 'never_submitted_retired':
        retirement = proof.get('retirement'); require(retirement in proof['evidence'], 'retirement evidence missing')
        retired = json.loads(Path(retirement['path']).read_text())
        require(retired.get('campaign_id') == root.name and retired.get('source_plan') == proof['source_plan']
                and value['unit_index'] in retired.get('retired_unit_ids', []), 'unit is not retired by cluster dispatcher')
        transition = proof.get('dispatcher_handoff'); require(transition in proof['evidence'], 'dispatcher handoff missing')
        state = json.loads(Path(transition['path']).read_text())
        require(state.get('old_owner_absent') is True and state.get('new_owner_job_id') and
                state.get('retirement') == retirement, 'dispatcher did not preserve exclusive retirement')
    else:
        require(proof.get('scheduler_never_started') is True and proof.get('terminal_state') == 'CANCELLED'
                and proof.get('native_output_absent') is True, 'cancelled unit has ambiguous execution')
    return {'path':str(p),'sha256':sha(p)}


def enqueue(job, root, handoff, capacity, gpu='RTX 4090'):
    job, root = Path(job).resolve(), Path(root).resolve(); value = verify(job, root)
    owner = validate_handoff(value, handoff, root)
    receipt = job/'vast_submission.json'
    if receipt.exists():
        old = json.loads(receipt.read_text()); require(old['handoff'] == owner and
            sha(root/'vast/queue'/(str(old['id'])+'.json')) == old['record_sha256'], 'contact queue changed')
        return str(old['id'])
    capacity_ref = validate_capacity(capacity, root)
    ids = root/'vast/queue/ids'; ids.mkdir(parents=True, exist_ok=True)
    jid = max([900000000]+[int(p.name) for p in ids.iterdir() if p.name.isdigit()])+1
    while True:
        try:(ids/str(jid)).mkdir();break
        except FileExistsError:jid+=1
    record = {'id':jid,'job_kind':'native_dependency','name':'isg261110_contact_map',
              'script':str(Path(value['helper_snapshot'])/'scripts/campaign_20261110/job_contact_map.sh'),
              'job_dir':str(job),'time_s':7500,'cpus':9,'mem':'48G','acct':'vision','part':'svl',
              'log':str(root/'slurm'/('contact_map_'+str(jid)+'.log')),'cwd':str(root),'submitted':time.time(),
              'submitted_by':'typed_contact_map','gpu':gpu,'native_inputs_sha256':sha(job/'native_inputs.json'),
              'exclusive_handoff':owner, 'cluster_first_observation':capacity_ref}
    target=root/'vast/queue'/(str(jid)+'.json');write(target,record)
    write(receipt,{'id':jid,'record_sha256':sha(target),'handoff':owner,'activated':False});return str(jid)


def validate_capacity(path, root):
    p = Path(path).resolve()
    require(p.is_relative_to(Path(root)/'infra/capacity_recovery'), 'cluster observation outside campaign')
    obs = json.loads(p.read_text())
    require(obs.get('campaign_id') == 'campaign_20261110' and obs.get('schema') == 'isg-contact-map-cluster-first-v1',
            'cluster observation identity differs')
    age = time.time() - obs['observed_unix']
    require(0 <= age <= 300, 'cluster-first observation stale')
    require(set(obs['accounts']) == {'minyeong', 'shiyuc'}, 'both cluster accounts must be observed')
    for account in obs['accounts'].values():
        require(account['returncode'] == 0 and account['pending_gpu_jobs'] > 0,
                'cluster independent pending overflow not verified')
        ref = account['evidence']; source = Path(ref['path']).resolve()
        require(source.is_relative_to(Path(root)) and sha(source) == ref['sha256'], 'cluster evidence changed')
    return {'path':str(p), 'sha256':sha(p)}


def preflight(job):
    value = verify(job); unit = value['unit']
    # Keep transport code separate from the original native implementation.
    sys.path.insert(0, value['snapshot'])
    from ispatialgym.sampling import contact_maps as cm
    require(Path(cm.__file__).resolve().is_relative_to(Path(value['snapshot'])), 'preflight loaded wrong native runtime')
    cfg = cm.contact_config(unit['map_kind'], lattice_m=.08, seeds=8,
                            **({'arm':unit['arm']} if unit['map_kind']=='wall_brace' else {}))
    require(json.loads(json.dumps(cfg.key())) == unit['expected_map_config'], 'native contact config differs')
    library = json.loads(Path(value['native_library']).read_text())
    entries = cm.select_entries(library, unit['map_kind'], ['valid'], arm=unit['arm'])
    require(len(entries) == unit['positive_entries'] and
            cm.pose_set(unit['map_kind'], entries, ['valid']) == unit['pose_set'], 'native positive pose set differs')
    result = {'verified':True, 'native_runtime':value['snapshot'], 'unit_index':value['unit_index'],
              'config':cfg.key(), 'pose_set':unit['pose_set'], 'benchmark_credit':0}
    write(Path(job)/'native_preflight.json', result)
    return result


def run(job):
    job=Path(job).resolve();value=verify(job);unit=value['unit'];out=job/'result';out.mkdir(exist_ok=False)
    argv=[sys.executable,'-u','-m','ispatialgym.sampling.contact_maps','compute','--root',value['private_contact_root'],
          '--kind',unit['map_kind'],'--only',unit['entry_id'],'--heights',unit['height_key'],
          '--verdicts','valid','--lattice','0.08','--seeds','8','--out',str(out/'maps')]
    # Fixed-arm kinds derive their arm from native semantics; per-entry kinds
    # must run only the requested arm instead of silently computing both.
    if unit['map_kind']=='wall_brace':argv+=['--arm',unit['arm']]
    start=time.time();record={'argv':argv,'started_at':start,'native_runtime':value['snapshot'],
        'unit_index':value['unit_index'],'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'input_manifest_sha256':sha(job/'native_inputs.json')};write(out/'started.json',record)
    pre = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--preflight', str(job)],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180)
    (out/'preflight.log').write_text(pre.stdout)
    if pre.returncode != 0:
        write(out/'contact_map_result.json', dict(record, finished_at=time.time(), returncode=pre.returncode,
              state='native_identity_preflight_failed', positive_native_reader_required=True, benchmark_credit=0))
        return 2
    with (out/'native.log').open('w') as log:
        try:rc=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=value['time_limit_s']).returncode
        except subprocess.TimeoutExpired:rc=124
    files={str(p):sha(p) for p in (out/'maps').rglob('cmap__*') if p.suffix in ('.json','.npz')}
    done='GB_CONTACT_MAPS_DONE' in (out/'native.log').read_text(errors='replace')
    record.update(finished_at=time.time(),returncode=rc,native_map_done=done,files=files,
        positive_native_reader_required=True,benchmark_credit=0)
    write(out/'contact_map_result.json',record)
    return 0 if rc==0 and done and files else 2


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify');p.add_argument('--run');p.add_argument('--preflight');a=p.parse_args()
    if a.verify:verify(a.verify);print('exact contact map dependency inputs verified')
    elif a.run:sys.exit(run(a.run))
    elif a.preflight:preflight(a.preflight)
    else:p.error('choose --verify or --run')
