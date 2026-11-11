"""Bounded real sampler A/B/A proof; independent episode gates remain required."""
from __future__ import annotations
import argparse
import contextlib
import datetime as dt
import json
import os
from pathlib import Path
import time
from scripts.campaign_20261110.contracts import file_digest, require, write_json_atomic
from scripts.campaign_20261110.training_native_canary import prepare


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sequence_inputs(root, requests, request_sha256):
    requests = Path(requests)
    require(file_digest(requests) == request_sha256, 'sequence request changed')
    document = json.loads(requests.read_text())
    require(document.get('sequence') == ['A', 'B', 'A'] and len(document['records']) == 3,
            'exact three-template A/B/A sequence required')
    prepared = [prepare(root, requests, i) for i in range(3)]
    rows = [p[0] for p in prepared]
    require(len({r['start'] for r in rows}) == 3, 'sequence indices overlap')
    require(len({r['native_runtime'] for r in rows}) == 1, 'sequence native runtimes differ')
    require(len({p[1]['dependency_release'] for p in prepared}) == 1, 'one shared exact dependency release required')
    environments = [{k:v for k,v in p[2].items() if k != 'ISPATIALGYM_CAMPAIGN_CONFIG'} for p in prepared]
    require(environments[0] == environments[1] == environments[2], 'sequence native environments differ')
    require(environments[0].get('ISPATIALGYM_CAMPAIGN_MODEL_POOL_MANIFEST'), 'explicit model pool required')
    identities = []
    for row, job, env, decision, argv, output in prepared:
        cfg = json.loads(Path(row['config_path']).read_text())
        targets = [o for o in cfg['template']['objects'] if o['role'] == 'target']
        require(len(targets) == 1, 'one exact target per bounded sequence member required')
        identities.append({k:targets[0][k] for k in ('category','model','default_scale')})
    require(identities[0] == identities[2] and identities[0] != identities[1], 'sequence target identity is not A/B/A')
    return prepared, identities


def parser_preflight(prepared):
    from types import SimpleNamespace
    from ispatialgym.sampling.v2.problem import GroundedProblem
    from ispatialgym.sampling.sample_isg import variant_split, world_key, stable_scene_json
    from ispatialgym.sampling import campaign_runtime
    from ispatialgym.sampling.model_pool import read_manifest
    keys = []
    for row, job, env, decision, argv, output in prepared:
        os.environ.update(env)
        gp = GroundedProblem.load(job['alias'], argv[argv.index('--customs')+1], argv[argv.index('--bddl-root')+1])
        variant_split(SimpleNamespace(seed=int(argv[argv.index('--seed')+1]), seed_index=0,
            seed_split=None, scene=job['scene'], room=None, scene_from_split=False), gp)
        require(gp.rooms(job['scene']) == [job['room']], 'native room binding differs')
        campaign_runtime.configuration()
        manifest, reference = read_manifest(env['ISPATIALGYM_CAMPAIGN_MODEL_POOL_MANIFEST'],
            scene=job['scene'], rooms=gp.rooms(job['scene']), fixed_bindings=gp.bindings(job['scene']))
        keys.append({'world_key':world_key(gp, job['scene'], stable_scene_json(job['scene'])),
                     'pool_manifest':reference})
    require(keys[0] == keys[1] == keys[2], 'native world/pool compatibility differs')
    return keys


def verify_stage(pool, original_identity, event_offset, target, output):
    require(pool.n_loads == 1 and pool.world is not None and pool.kit is not None, 'sampler world rebuilt or missing')
    model_pool = pool.world.model_pool
    identity = (id(pool.world), id(pool.kit), id(model_pool),
                tuple(sorted((name,id(obj)) for name,obj in model_pool.members.items())))
    require(original_identity is None or identity == original_identity, 'preloaded world/object identity changed')
    require(not pool.world.spawned and not model_pool.active, 'ordinary template cleanup did not finish')
    events = model_pool.events[event_offset:]
    exports = [e for e in events if e['event'] == 'exported']
    require(exports and events[-1]['event'] == 'released', 'native export/cleanup evidence absent')
    active_names = set()
    for event in exports:
        observation = event['native_observation']
        require(observation['pool_physics_rebuilds'] == 1, 'pool physical inventory rebuilt')
        active = set(event['active']); require(len(active) == 1, 'wrong active task inventory')
        active_names |= active
        for name, obj in observation['objects'].items():
            if name in active:
                require(obj['category'] == target['category'] and obj['model'] == target['model'] and obj['visible'],
                        'wrong or invisible active target')
            else:
                require(not obj['visible'] and not obj['planner_included'] and not any(obj['collision_enabled']),
                        'inactive model still participates in scene')
    exports_on_disk = list(Path(output).rglob('template-partial_rooms.json'))
    require(len(exports_on_disk) == 1, 'one native template export required')
    scene = json.loads(exports_on_disk[0].read_text())
    meta = scene['metadata']['ispatialgym']['model_pool']
    require(set(meta['active_names']) == active_names and meta['manifest_sha256'] == pool.model_pool_key,
            'native export pool identity differs')
    loaded = set(model_pool.members); initial = set(scene['objects_info']['init_info'])
    require(initial & loaded == active_names, 'inactive model leaked into export')
    return identity, {'world_loads':pool.n_loads,'same_world_and_members':True,
        'ordinary_cleanup_complete':True,'active_names':sorted(active_names),
        'inactive_export_omissions':len(loaded-active_names), 'pool_execution':{'path':str(model_pool.output),
        'sha256':file_digest(model_pool.output)},'template_export':{'path':str(exports_on_disk[0]),
        'sha256':file_digest(exports_on_disk[0])}, 'event_range':[event_offset,len(model_pool.events)],
        'scope':'Sampler physical isolation/export only; independent demo, replay and render gates remain separate.'}


def run(root, requests, request_sha256, *, preflight=False):
    requests = Path(requests).resolve(); destination = requests.parent/'sequence_result.json'
    prepared, targets = sequence_inputs(root, requests, request_sha256)
    keys = parser_preflight(prepared)
    proof = {'campaign_id':'campaign_20261110','observed_at':now(), 'request':{'path':str(requests),'sha256':request_sha256},
             'sequence':['A','B','A'],'native_parser_passed':True,'compatible_worlds':keys,
             'benchmark_credit':0,'production_released':False,'stages':[],'status':'preflight_passed'}
    if preflight:
        write_json_atomic(requests.parent/'sequence_preflight.json',proof);return proof
    require(not destination.exists(), 'preserve previous sequence execution')
    from ispatialgym.sampling import sample_isg
    import omnigibson as og
    require(Path(sample_isg.__file__).resolve().is_relative_to(Path(prepared[0][0]['native_runtime']).resolve()),
            'physical sampler import differs from pinned runtime')
    pool = None; original_identity = None
    try:
        for index, target in enumerate(targets):
            # Recheck current policy and all immutable inputs immediately before each native member.
            row,job,env,decision,argv,output = prepare(root, requests, index)
            os.environ.update(env);output.mkdir(parents=True)
            command = {'campaign_id':'campaign_20261110','candidate_id':row['candidate_id'],
                'request_sha256':request_sha256,'config_sha256':row['config_sha256'],'gate':decision,
                'argv':argv,'native_runtime':row['native_runtime'],'created_at':now(),
                'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'production_released':False,'eligible_accepted':0,
                'sequence_position':index,'borrowed_pool':pool is not None}
            write_json_atomic(output/'command.json',command)
            offset = len(pool.world.model_pool.events) if pool else 0
            started = time.monotonic();rc = 2;failure = None;isolation = None
            try:
                with (output/'sample.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    rc,report,pool = sample_isg.main(argv[3:],shutdown=False,existing_pool=pool)
                require(rc == 0 and report['n_written'] == 1, 'bounded native member did not write exactly one TRO')
                original_identity,isolation = verify_stage(pool,original_identity,offset,target,output)
            except Exception as exc:
                failure=repr(exc);rc=2
            tros=list(output.rglob('*-tro_state.json'))
            result={**command,'returncode':rc,'tro_count':len(tros),
                'tros':[{'path':str(p),'sha256':file_digest(p)}for p in tros],
                'elapsed_seconds':time.monotonic()-started,'finished_at':now(),
                'error':failure,'sampler_pool_isolation':isolation,'benchmark_credit':0}
            write_json_atomic(output/'sampling_gate.json',result)
            proof['stages'].append({'candidate_id':row['candidate_id'],'start':row['start'],
                'native_gate':{'path':str(output/'sampling_gate.json'),'sha256':file_digest(output/'sampling_gate.json')},
                'native_passed':rc==0 and len(tros)==1,'isolation':isolation,'error':failure})
            proof.update(status='running' if rc==0 else 'failed',observed_at=now())
            write_json_atomic(destination,proof)
            require(rc==0 and len(tros)==1, 'sequence stopped after preserved failed member: '+str(failure))
        proof.update(status='sampler_sequence_passed',observed_at=now(),next_required=['each exact template demo','each own physics replay','each rendering proof'])
        write_json_atomic(destination,proof)
    finally:
        og.shutdown()
    return proof


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--requests',type=Path,required=True);p.add_argument('--request-sha256',required=True)
    p.add_argument('--preflight',action='store_true');a=p.parse_args()
    print(json.dumps(run(a.root,a.requests,a.request_sha256,preflight=a.preflight)))
