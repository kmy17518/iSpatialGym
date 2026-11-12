"""Prepare required object-independent generic maps; never submit or sample scenes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from scripts.campaign_20261110.prepare_consumed_attachment_maps import support_identity_matches


def identity(height, pose_set, config, robot, policy, generator):
    return {'kind': 'generic', 'height': float(height), 'arm': 'right', 'pose_set': pose_set,
            'config': config, 'robot': robot, 'policy': policy, 'generator_files': generator}


def prepare(root, runtime, graph_path, geometry_path, rests_path, output):
    from ispatialgym.grasping import contact_policy as CP
    from ispatialgym.sampling import reach_maps as RM
    root, runtime, graph_path, geometry_path, rests_path, output = map(Path, (root, runtime, graph_path, geometry_path, rests_path, output))
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    ref = lambda p: {'path': str(p), 'sha256': sha(p)}
    read = lambda p: json.loads(Path(p).read_text())
    assert not output.exists()
    graph, geometry, rests = read(graph_path), read(geometry_path), read(rests_path)
    # Initial measured source has an upright designated rest. Other object poses
    # must supply their own proven rotated hull height before joining this key.
    selected_identity = (geometry['category'], geometry['model'], geometry['scale'])
    rest_rows = rests.get('rests', rests)
    if isinstance(rest_rows, dict): rest_rows = list(rest_rows.values())
    rest = next(r for r in rest_rows if r['rest_id'] == 'r0')
    assert rest['stable'] and rest['contact'] and rest['scale'] == geometry['scale']
    quat = rest['quat_xyzw']
    assert sum(float(v) ** 2 for v in quat[:3]) < 1e-8 and abs(float(quat[3])) > .9999
    extent = float(geometry['coll_bbox_max'][2]) - float(geometry['coll_bbox_min'][2])
    pose = RM.generic_pose_set(RM.generic_dz(extent))
    cfg = json.loads(json.dumps(RM.MapConfig(arm='right', twins=False).key()))
    generator = {n: sha(runtime / n) for n in ('ispatialgym/sampling/reach_maps.py', 'ispatialgym/grasping/contact_policy.py',
                 'ispatialgym/robot/conventions.py', 'ispatialgym/sampling/reach_bands.py')}
    supports = {}
    for row in graph['rows']:
        if row['kind'] == 'native_support_geometry_and_height' and row['status'] == 'verified':
            proof = row['artifact']; assert sha(proof['path']) == proof['sha256']
            supports[(row['inputs']['support_profile'], row['inputs']['height_slot'])] = (read(proof['path']), proof)
    grouped, blocked = {}, []
    for row in graph['rows']:
        if row['kind'] != 'generic_reach_map' or row['inputs']['arm'] != 'right': continue
        target = row['inputs']['target']
        if (target['category'], target['model'], target.get('applied_scale')) != selected_identity: continue
        selected = row['inputs']['support']; pair = supports.get((selected['support_profile'], selected['height_slot']))
        reason = 'support_height_missing' if pair is None else 'support_instance_identity_mismatch' if not support_identity_matches(selected, pair[0]) else None
        if reason:
            blocked.append({'dependency_id': row['id'], 'configuration_ids': row['consumer_scopes']['required'], 'blocker': reason, 'support': selected}); continue
        measured, proof = pair; assert measured['geometry_admitted']
        unit = grouped.setdefault(measured['actual_height_key'], {'proofs': {}, 'configuration_ids': set(), 'dependency_ids': set(), 'class_ids': set()})
        unit['proofs'][proof['path']] = proof; unit['configuration_ids'].update(row['consumer_scopes']['required'])
        unit['dependency_ids'].add(row['id']); unit['class_ids'].update(row['required_by_classes'])
    # Exact cache candidates are checked independently per height. Never wait on
    # unrelated models, and never turn a cache hit into another GPU computation.
    directories = list(root.glob('v2_release*/maps/_generic'))
    directories += [root.parent / 'oracle_fix/v2maps_int2/maps/_generic']
    for base in (root / 'training_preparation', root / 'out'):
        for child in base.iterdir():
            if 'generic' in child.name and child.is_dir(): directories.extend(child.rglob('_generic'))
    output.mkdir(parents=True); tickets = []; cached = []
    namespace = root / 'infra/coordination/preparation_dispatch/consumed_asset_builds_r1'
    for height, unit in sorted(grouped.items()):
        physical = identity(height, pose, cfg, RM.robot_digest(), CP.DEFAULT.key('right'), generator)
        key = hashlib.sha256(json.dumps(physical, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        hits = []
        for directory in directories:
            hit = RM.find_generic(str(directory), float(height), arm='right', pose_set=pose,
                                  policy_key=CP.DEFAULT.key('right'), robot=RM.robot_digest(), config_require=cfg)
            if hit is not None: hits.append(ref(hit.path))
        if hits:
            cached.append({'physical_dedup_key': key, 'height': height, 'candidates': hits, 'status': 'native_lookup_hit_pending_grid_audit'}); continue
        assert not (namespace / 'claims' / (key + '.json')).exists(), 'Physical key already claimed'
        ident = 'generic_right_' + key[:20]; directory = output / ident; directory.mkdir()
        jobs = directory / 'jobs.json'; jobs.write_text(json.dumps([{'id': '_generic', 'kind': 'generic', 'dz_m': pose['dz_m']}], indent=2) + '\n')
        h = float(height); anchors = directory / 'anchors.json'
        anchors.write_text(json.dumps({'schema': 'isg-height-anchors/1', 'anchors': [{'height': h, 'key': height, 'precision': .0001,
            'range': [h - .00005, h + .00005], 'status': ['measured'], 'surfaces': sorted(unit['proofs'])}], 'max_approx_m': .00005}, indent=2) + '\n')
        selected = directory / 'consumed_inputs.json'
        selected.write_text(json.dumps({'physical_identity': physical, 'source_graph': ref(graph_path), 'geometry': ref(geometry_path),
            'rests': ref(rests_path), 'object_height': extent, 'designated_rest_id': 'r0', 'height_proofs': list(unit['proofs'].values()),
            'configuration_ids': sorted(unit['configuration_ids']), 'dependency_ids': sorted(unit['dependency_ids']),
            'requires_grasp_library': False, 'requires_scene_query': False}, indent=2) + '\n')
        inputs = [ref(p) for p in (jobs, anchors, selected, geometry_path, rests_path, runtime / '.snapshot_commit')]
        inputs += list(unit['proofs'].values()) + [{'path': str(runtime / n), 'sha256': h} for n, h in generator.items()]
        argv = ['python', '-u', '-m', 'ispatialgym.sampling.reach_maps', 'compute', '--kind', 'generic', '--jobs', str(jobs),
                '--anchors', str(anchors), '--heights', height, '--arm', 'right', '--out', str(directory / 'maps'),
                '--mode', 'full', '--lattice', '0.08', '--seeds', '8', '--trunk-eligibility', 'per_d_e']
        prepared = directory / 'prepared.json'
        prepared.write_text(json.dumps({'id': ident, 'physical_dedup_key': key, 'physical_identity': physical, 'input_refs': inputs,
            'native_runtime': str(runtime), 'native_argv': argv, 'configuration_ids': sorted(unit['configuration_ids']), 'benchmark_credit': 0}, indent=2) + '\n')
        runner = '''import hashlib,json,os,subprocess,sys,time
from pathlib import Path
D=Path(__file__).parent;p=json.loads((D/'prepared.json').read_text());sha=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
for r in p['input_refs']:assert sha(r['path'])==r['sha256'],r['path']
a=p['native_argv'];a[0]=sys.executable;v={'job_id':os.environ['SLURM_JOB_ID'],'physical_dedup_key':p['physical_dedup_key'],'prepared_sha256':sha(D/'prepared.json'),'started_at':time.time(),'benchmark_credit':0}
(D/'started.json').write_text(json.dumps(v,indent=2))
with(D/'native.log').open('w')as f:
 try:rc=subprocess.run(a,stdout=f,stderr=subprocess.STDOUT,timeout=1500).returncode
 except subprocess.TimeoutExpired:rc=124
v.update(returncode=rc,finished_at=time.time(),native_done='GB_MAP_DONE'in(D/'native.log').read_text(errors='replace'),map_files=[{'path':str(x),'sha256':sha(x)}for x in(D/'maps').rglob('map__*')if x.is_file()])
(D/'terminal.json').write_text(json.dumps(v,indent=2)+'\\n');sys.exit(0 if rc==0 and v['native_done']and v['map_files']else 2)
'''
        compile(runner, 'run.py', 'exec'); (directory / 'run.py').write_text(runner)
        env = {'ISG_GB_ROOT': str(runtime), 'GBH': str(root), 'KIT_TPL_NFS_ROOT': str(root / 'kit_templates'),
               'KIT_TEX_STORE': str(root / 'kit_textures'), 'EXP_PATH': str(root / 'isaacsim_overlay/apps')}
        shell = '#!/bin/bash\nset -euo pipefail\n' + '\n'.join('export ' + k + '=' + shlex.quote(v) for k, v in env.items())
        shell += '\nset +u\nsource "$ISG_GB_ROOT/scripts/grasp_bands/env_cluster.sh"\nset -u\nexport PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR="$JB/numba_cache"\nKIT_RC=1\ntrap \'kit_appdata_teardown\' EXIT\npython -u ' + shlex.quote(str(directory / 'run.py')) + '\nKIT_RC=0\n'
        (directory / 'job.sh').write_text(shell); (directory / 'pinned.sh').write_text('#!/bin/bash\nset -euo pipefail\nexec srun --cpu-bind=cores --exact -n1 -c8 bash ' + shlex.quote(str(directory / 'job.sh')) + '\n')
        subprocess.run(['bash', '-n', str(directory / 'job.sh')], check=True)
        tickets.append({'id': ident, 'physical_dedup_key': key, 'dedup_key': key, 'expected_account': 'minyeong', 'source_account': 'minyeong',
            'claim_namespace': str(namespace / 'claims'), 'receipt_namespace': str(namespace / 'receipts'),
            'claim_path': str(namespace / 'claims' / (key + '.json')), 'submission_path': str(namespace / 'receipts' / (key + '.json')),
            'sbatch_argv': ['sbatch', '--parsable', '--nodes=1', '--ntasks=1', '-A', 'vision', '-p', 'svl,viscam', '--exclude=svl9', '--gres=gpu:titanrtx:1', '-c', '8', '--mem=48G', '--time=00:30:00', '--nice=1000', '-J', 'isg261110_generic_maps', '-o', str(directory / 'slurm_%j.log'), str(directory / 'pinned.sh')],
            'prepared': ref(prepared), 'runtime_ref': ref(runtime / '.snapshot_commit'), 'input_refs': inputs,
            'scripts': [ref(directory / n) for n in ('run.py', 'job.sh', 'pinned.sh')], 'output': str(directory / 'maps'),
            'dispatch_workdir': str(runtime), 'hard_time_seconds': 1800, 'native_time_seconds': 1500,
            'computation_kind': 'reach_map', 'consumer_kind': 'generic', 'workstream': 'training preparation', 'owner': '/root/templates_assets',
            'stage': 'required generic map', 'class_ids': sorted(unit['class_ids']), 'configuration_ids': sorted(unit['configuration_ids']),
            'selected_dependency_ids': sorted(unit['dependency_ids']), 'ready_since': time.time(),
            'followup': 'Strict native generic reader: exact pose-set, height, arm, policy, robot, config, finite grid, status coverage and consumer lookup.',
            'benchmark_credit': 0, 'production_authorized': False})
    manifest = output / 'tickets.json'; manifest.write_text(json.dumps({'schema': 'isg-preparation-dispatch-tickets-v1', 'campaign_id': root.name,
        'account': 'minyeong', 'source_account': 'minyeong', 'claim_namespace': str(namespace / 'claims'), 'receipt_namespace': str(namespace / 'receipts'),
        'observed_at': time.time(), 'owner': '/root/templates_assets', 'dispatch_owner': '/root', 'jobs': tickets,
        'cached': cached, 'blocked': blocked, 'source_graph': ref(graph_path), 'benchmark_credit': 0,
        'production_authorized': False, 'full_inventory_complete': False}, indent=2) + '\n')
    print(json.dumps({'manifest': ref(manifest), 'ready': len(tickets), 'cached': len(cached), 'blocked': len(blocked)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'runtime', 'graph', 'geometry', 'rests', 'output'): parser.add_argument('--' + name, required=True)
    args = parser.parse_args(); prepare(args.root, args.runtime, args.graph, args.geometry, args.rests, args.output)
