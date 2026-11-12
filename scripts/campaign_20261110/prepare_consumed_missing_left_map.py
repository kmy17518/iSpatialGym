"""Isolate one missing LEFT map after an evidenced native bootstrap failure."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


def main(account):
    import numpy as np
    from ispatialgym.grasping import schema as S, contact_policy as CP
    from ispatialgym.sampling import reach_maps as RM
    R = Path('/vision/u/minyeong/projects/isg_home/campaign_20261110')
    root = R if account == 'minyeong' else Path('/vision/u/shiyuc/minyeong/campaign_20261110')
    runtime = R / 'code/trunk_3908643f36d0'
    batch, index, revision = (53, 4, 1) if account == 'minyeong' else (4, 3, 2)
    name = 'selected_all_verified_batch_%03d_maps_dispatch_r1' % batch
    build = root / 'training_preparation' / name / 'build_dispatch_r1'; old = build / 'units' / str(index)
    proof_path = R / ('training_preparation/selected_partial_reach_reader_r%d' % revision) / account / name / 'units' / ('%03d.json' % index)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    ref = lambda p: {'path': str(p), 'sha256': sha(p)}
    read = lambda p: json.loads(Path(p).read_text())
    proof = read(proof_path); assert proof['status'] == 'verified_partial_arm' and proof['remaining_arms'] == ['left']
    plan_ref = proof['source_plan']; assert sha(plan_ref['path']) == plan_ref['sha256']
    plan = read(plan_ref['path']); unit = next(x for x in plan['units'] if x['id'] == index); job = unit['job']
    terminal = read(old / 'terminal.json'); assert terminal['native_map_done'] is False and terminal['native_runtime'] == str(runtime)
    failure_log = old / 'logs/maps_0_left.log'; text = failure_log.read_text(errors='replace')
    assert 'GB_MAP_DONE' not in text
    cause = 'SimulationApp viewport segmentation fault/abort' if account == 'minyeong' else 'SimulationApp missing omni.kit.usd extension'
    assert ('Fatal Python error: Segmentation fault' in text if account == 'minyeong' else "No module named 'omni.kit.usd'" in text)
    libpath = build / 'glib/models' / unit['entry_id'] / 'library.json'; assert sha(libpath) == unit['source_library']['sha256']
    lib = S.load_model(job['category'], job['model'], job['scale'], str(build / 'glib'), constraint=job['constraint'])
    assert lib is not None and len(lib.grasps()) > 0
    for record in proof['maps']:
        for r in (record, record['npz']): assert sha(r['path']) == r['sha256']
        native = RM.ReachMap(record['path']); meta = native.meta
        assert meta['key'] == RM.map_key(meta) and meta['object_key'] == lib.identity_key() and meta['arm'] == 'right'
        assert np.isfinite(native.cells).all() and np.isfinite(native.yaws).all()
        statuses = [v for k, v in native.arrays.items() if k.startswith('s')]
        assert statuses and sum(int((v == RM.STATUS['feasible']).sum()) for v in statuses) == record['positive_cells'] > 0
    source_refs = [ref(runtime / name) for name in ('ispatialgym/sampling/reach_maps.py', 'ispatialgym/grasping/schema.py',
                   'ispatialgym/grasping/contact_policy.py', 'ispatialgym/grasping/hand_geometry.py', 'ispatialgym/grasping/object_geometry.py')]
    identity = {'kind': 'tabletop', 'library_identity': lib.identity_key(), 'library_sha256': sha(libpath), 'arm': 'left',
                'height': unit['height'], 'rest_ids': sorted({g['rest_id'] for g in lib.grasps()}),
                'robot': RM.robot_digest(), 'policy': CP.DEFAULT.key('left'), 'config': RM.MapConfig(arm='left').key(),
                'generator_sha256': [r['sha256'] for r in source_refs]}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    D = root / 'training_preparation/consumed_missing_left_maps_r1' / ('left_' + key[:20]); assert not D.exists(); D.mkdir(parents=True)
    inputs = source_refs + [ref(runtime / '.snapshot_commit'), ref(proof_path), ref(old / 'terminal.json'), ref(failure_log),
                             ref(old / 'jobs.json'), ref(old / 'anchors.json'), plan_ref]
    inputs += [ref(p) for p in sorted(libpath.parent.iterdir()) if p.is_file()]
    argv = ['python', '-u', '-m', 'ispatialgym.sampling.reach_maps', 'compute', '--kind', 'tabletop', '--jobs', str(old / 'jobs.json'),
            '--glib', str(build / 'glib'), '--anchors', str(old / 'anchors.json'), '--heights', unit['height'], '--arm', 'left',
            '--shard', '0/1', '--out', str(D / 'maps'), '--mode', 'full', '--lattice', '0.08', '--seeds', '8', '--trunk-eligibility', 'per_d_e']
    prepared = {'id': D.name, 'physical_dedup_key': key, 'physical_identity': identity, 'input_refs': inputs, 'native_argv': argv,
                'native_runtime': str(runtime), 'prior_job_id': terminal['job_id'], 'failure_cause': cause,
                'changed_execution': 'LEFT-only process with a fresh job-specific Kit/appdata and cache namespace; strict completion marker required; RIGHT output and old claims retained.',
                'right_map_proof': ref(proof_path), 'original_claim_preserved': str(build / 'preparation.claim'),
                'benchmark_credit': 0, 'production_authorized': False}
    pp = D / 'prepared.json'; pp.write_text(json.dumps(prepared, indent=2) + '\n')
    runner = '''import hashlib,json,os,subprocess,sys,time
from pathlib import Path
D=Path(__file__).parent;p=json.loads((D/'prepared.json').read_text());sha=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
for r in p['input_refs']:assert sha(r['path'])==r['sha256'],r['path']
argv=list(p['native_argv']);argv[0]=sys.executable;v={'job_id':os.environ['SLURM_JOB_ID'],'started_at':time.time(),'prepared_sha256':sha(D/'prepared.json'),'physical_dedup_key':p['physical_dedup_key'],'native_runtime':p['native_runtime'],'arm':'left','benchmark_credit':0}
(D/'started.json').write_text(json.dumps(v,indent=2)+'\\n')
with(D/'native_left.log').open('w')as log:
 try:rc=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=1500).returncode
 except subprocess.TimeoutExpired:rc=124
done='GB_MAP_DONE'in(D/'native_left.log').read_text(errors='replace');v.update(returncode=rc,native_done=done,finished_at=time.time(),map_files=[{'path':str(x),'sha256':sha(x)}for x in(D/'maps').rglob('map__*')if x.is_file()],native_reader_required=True)
(D/'terminal.json').write_text(json.dumps(v,indent=2)+'\\n');sys.exit(0 if rc==0 and done and v['map_files']else 2)
'''
    compile(runner, 'run.py', 'exec'); (D / 'run.py').write_text(runner)
    env = {'ISG_GB_ROOT': str(runtime), 'GBH': str(root), 'KIT_TPL_NFS_ROOT': str(root / 'kit_templates'),
           'KIT_TEX_STORE': str(root / 'kit_textures'), 'EXP_PATH': str(root / 'isaacsim_overlay/apps')}
    shell = '#!/bin/bash\nset -euo pipefail\n' + '\n'.join('export ' + k + '=' + shlex.quote(v) for k, v in env.items())
    shell += '\nset +u\nsource "$ISG_GB_ROOT/scripts/grasp_bands/env_cluster.sh"\nset -u\nexport PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR="$JB/numba_cache"\nKIT_RC=1\ntrap \'kit_appdata_teardown\' EXIT\npython -u ' + shlex.quote(str(D / 'run.py')) + '\nKIT_RC=0\n'
    (D / 'job.sh').write_text(shell); (D / 'pinned.sh').write_text('#!/bin/bash\nset -euo pipefail\nexec srun --cpu-bind=cores --exact -n1 -c8 bash ' + shlex.quote(str(D / 'job.sh')) + '\n')
    subprocess.run(['bash', '-n', str(D / 'job.sh')], check=True)
    source_consumers = read(R / 'training_preparation/model_variant_inventory_r2/consumed_scope_r1/maps1050_incomplete_consumer_audit_r1.json')
    consumers = next(row for row in source_consumers['rows'] if row['wave_unit_id'] == unit['wave_unit_id'])
    graph = read(R / 'training_preparation/model_variant_inventory_r2/consumed_scope_r1/dependencies.json')
    left_requirements = [r['id'] for r in graph['rows'] if r['id'] in consumers['requirement_ids'] and r['inputs']['arm'] == 'left']
    assert left_requirements
    namespace = R / 'infra/coordination/preparation_dispatch/consumed_asset_builds_r1'
    ticket = {'id': D.name, 'physical_dedup_key': key, 'dedup_key': key, 'expected_account': account,
              'claim_path': str(namespace / 'claims' / (key + '.json')), 'submission_path': str(namespace / 'receipts' / (key + '.json')),
              'sbatch_argv': ['sbatch', '--parsable', '--nodes=1', '--ntasks=1', '-A', 'vision', '-p', 'svl,viscam', '--exclude=svl9', '--gres=gpu:titanrtx:1', '-c', '8', '--mem=48G', '--time=00:30:00', '--nice=1000', '-J', 'isg261110_missing_left_map', '-o', str(D / 'slurm_%j.log'), str(D / 'pinned.sh')],
              'prepared': ref(pp), 'runtime_ref': ref(runtime / '.snapshot_commit'), 'input_refs': inputs, 'scripts': [ref(D / n) for n in ('run.py', 'job.sh', 'pinned.sh')],
              'output': str(D / 'maps'), 'dispatch_workdir': str(runtime), 'hard_time_seconds': 1800, 'native_time_seconds': 1500,
              'computation_kind': 'reach_map', 'consumer_kind': 'tabletop', 'workstream': 'training preparation', 'owner': '/root/templates_assets',
              'stage': 'missing LEFT map after native bootstrap failure', 'configuration_ids': consumers['configuration_ids'],
              'class_ids': consumers['class_ids'], 'selected_dependency_ids': left_requirements,
              'original_claim_preserved': prepared['original_claim_preserved'], 'failure_terminal': ref(old / 'terminal.json'),
              'benchmark_credit': 0, 'production_authorized': False}
    out = D.parent / ('tickets_' + account + '.json'); assert not out.exists()
    out.write_text(json.dumps({'schema': 'isg-preparation-dispatch-tickets-v1', 'campaign_id': R.name, 'account': account, 'jobs': [ticket],
                              'owner': '/root/templates_assets', 'dispatch_owner': '/root', 'observed_at': time.time(), 'benchmark_credit': 0,
                              'production_authorized': False}, indent=2) + '\n')
    print(json.dumps({'manifest': ref(out), 'preserved_right_positive_cells': sum(m['positive_cells'] for m in proof['maps']), 'cause': cause}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--account', choices=['minyeong', 'shiyuc'], required=True)
    main(parser.parse_args().account)
