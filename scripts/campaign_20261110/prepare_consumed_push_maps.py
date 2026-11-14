"""Prepare exact required RIGHT axial-push maps from verified contact libraries."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from scripts.campaign_20261110.prepare_consumed_attachment_maps import support_identity_matches


def selected_dependency(dep, obj):
    target = dep.get('inputs', {}).get('target', {})
    return (dep.get('kind') == 'contact_push_map' and dep['inputs'].get('arm') == 'right'
            and all(c.startswith('alignment-axial_') for c in dep['required_by_classes'])
            and target.get('category') == obj['category'] and target.get('model') == obj['model']
            and target.get('applied_scale') == obj['scale'])


def prepare(root, runtime, audit_path, cache_path, graph_path, output):
    root, runtime, audit_path, cache_path, graph_path, output = map(Path, (root, runtime, audit_path, cache_path, graph_path, output))
    sys.path.insert(0, str(runtime))
    from ispatialgym.grasping import contact_library as CL, contact_policy as CP
    from ispatialgym.sampling import contact_maps as CM, reach_maps as RM
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    ref = lambda p: {'path': str(p), 'sha256': sha(p)}
    read = lambda p: json.loads(Path(p).read_text())
    def write(p, value):
        with p.open('x') as stream: stream.write(json.dumps(value, indent=2) + '\n')
    assert not output.exists(); output.mkdir(parents=True)
    audit, graph, cached = map(read, (audit_path, graph_path, cache_path))
    assert not cached['errors']
    owned = [u for r in cached['requests'] for u in read(r['file']['path']).get('units', [])]
    for row in cached['requests']: assert sha(row['file']['path']) == row['file']['sha256']
    supports = {}
    for row in graph['rows']:
        if row['kind'] == 'native_support_geometry_and_height' and row['status'] == 'verified':
            proof = row['artifact']; assert sha(proof['path']) == proof['sha256']
            supports[(row['inputs']['support_profile'], row['inputs']['height_slot'])] = (proof, read(proof['path']))
    namespace = root / 'infra/coordination/preparation_dispatch/consumed_asset_builds_r1'
    generator = {name: sha(runtime / name) for name in ['ispatialgym/sampling/contact_maps.py', 'ispatialgym/sampling/reach_maps.py',
                 'ispatialgym/grasping/contact_library.py', 'ispatialgym/grasping/contact_policy.py', 'ispatialgym/sampling/v2/contact_poses.py']}
    runner = '''from pathlib import Path
import hashlib,json,os,subprocess,sys,time
D=Path(__file__).parent;p=json.loads((D/'prepared.json').read_text());sha=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
for r in p['input_refs']:assert sha(r['path'])==r['sha256'],r['path']
argv=list(p['native_argv']);argv[0]=sys.executable
v={'started_at':time.time(),'job_id':os.environ['SLURM_JOB_ID'],'physical_dedup_key':p['physical_dedup_key'],'prepared_sha256':sha(D/'prepared.json'),'native_runtime':p['native_runtime'],'benchmark_credit':0}
(D/'started.json').write_text(json.dumps(v,indent=2)+'\\n')
with(D/'native.log').open('w')as log:
 try:rc=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=6900).returncode
 except subprocess.TimeoutExpired:rc=124
maps=[{'path':str(f),'sha256':sha(f)}for f in(D/'maps').rglob('cmap__*')if f.suffix in('.json','.npz')]
v.update(finished_at=time.time(),returncode=rc,map_files=maps,native_done='GB_CONTACT_MAPS_DONE'in(D/'native.log').read_text(errors='replace'),production_authorized=False)
(D/'terminal.json').write_text(json.dumps(v,indent=2)+'\\n');sys.exit(0 if rc==0 and v['native_done'] and maps else 2)
'''
    compile(runner, 'run.py', 'exec'); tickets = []; retained = []; missing = Counter()
    for row in audit['rows']:
        if row['status'] != 'verified_native_contact_physics': continue
        obj = row['object']; library = Path(row['library']['path'])
        assert sha(library) == row['library']['sha256']; lib = CL.load_library(str(library))
        assert lib['digest'] == CL._library_digest(lib)
        entries = CM.select_entries(lib, 'push_blade', ('valid',), arm='right'); assert entries
        pose_set = CM.pose_set('push_blade', entries, ('valid',))
        assert any(s['map_kind'] == 'push_blade' and s['map_pose_set'] == pose_set for s in row['map_scopes'])
        grouped = {}
        for dep in graph['rows']:
            if not selected_dependency(dep, obj): continue
            support = dep['inputs']['support']; measured = supports.get((support['support_profile'], support['height_slot']))
            if measured is None: missing['support_height_missing'] += 1; continue
            proof, geometry = measured
            if not support_identity_matches(support, geometry): missing['support_geometry_identity_unresolved'] += 1; continue
            grouped.setdefault(geometry['actual_height_key'], []).append((dep, proof))
        prep = read(row['prepared']['path']); assert sha(row['prepared']['path']) == row['prepared']['sha256']
        for height, selected in sorted(grouped.items()):
            config = CM.contact_config('push_blade', arm='right', lattice_m=.08, seeds=8).key()
            physical = {'kind': 'push_blade', 'arm': 'right', 'height_above_floor': float(height), 'object_key': CM.object_key(obj),
                        'pose_set': pose_set, 'config': config, 'robot': RM.robot_digest(), 'policy_key': CP.DEFAULT.key('right'),
                        'generator_files': generator}
            key = hashlib.sha256(json.dumps(physical, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            existing = [u for u in owned if u.get('map_kind') == 'push_blade' and u.get('entry_id') == obj['entry']
                        and u.get('pose_set') == pose_set and u.get('height') == float(height) and u.get('expected_map_config') == config]
            maps = [m['file'] for m in cached['maps'] if m['metadata'].get('kind') == 'push_blade'
                    and m['metadata']['object']['entry'] == obj['entry'] and m['metadata']['pose_set'] == pose_set
                    and m['metadata']['height']['value'] == float(height)]
            if existing or maps or (namespace / 'claims' / (key + '.json')).exists():
                retained.append({'physical_dedup_key': key, 'existing_units': existing, 'cached_maps': maps}); continue
            ident = 'required_push_map_' + key[:20]; directory = output / ident; directory.mkdir()
            inputs = [ref(audit_path), ref(cache_path), ref(graph_path), row['library'], row['prepared'], row['terminal'], ref(runtime / '.snapshot_commit')]
            inputs += [{'path': p, 'sha256': h} for p, h in prep['private_model_files'].items()]
            inputs += [proof for dep, proof in selected] + [{'path': str(runtime / n), 'sha256': h} for n, h in generator.items()]
            inputs = list({r['path']: r for r in inputs}.values())
            for r in inputs: assert sha(r['path']) == r['sha256']
            configs = sorted({c for dep, _ in selected for c in dep['consumer_scopes']['required']})
            classes = sorted({c for dep, _ in selected for c in dep['required_by_classes']})
            argv = ['python', '-u', '-m', 'ispatialgym.sampling.contact_maps', 'compute', '--root', str(library.parents[3]),
                    '--kind', 'push_blade', '--arm', 'right', '--only', obj['entry'], '--heights', height, '--verdicts', 'valid',
                    '--lattice', '0.08', '--seeds', '8', '--out', str(directory / 'maps')]
            pp = directory / 'prepared.json'; write(pp, {'physical_dedup_key': key, 'physical_identity': physical, 'native_runtime': str(runtime),
                'native_argv': argv, 'source_library': row['library'], 'object': obj, 'pose_set': pose_set, 'height': float(height),
                'expected_map_config': config, 'input_refs': inputs, 'configuration_ids': configs, 'production_authorized': False, 'benchmark_credit': 0})
            (directory / 'run.py').write_text(runner)
            env = {'ISG_GB_ROOT': str(runtime), 'GBH': str(root), 'KIT_TPL_NFS_ROOT': str(root / 'kit_templates'),
                   'KIT_TEX_STORE': str(root / 'kit_textures'), 'EXP_PATH': str(root / 'isaacsim_overlay/apps')}
            shell = '#!/bin/bash\nset -euo pipefail\n' + '\n'.join('export ' + k + '=' + shlex.quote(v) for k, v in env.items())
            shell += '\nset +u\nsource "$ISG_GB_ROOT/scripts/grasp_bands/env_cluster.sh"\nset -u\nexport PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR="$JB/numba_cache"\nKIT_RC=1\ntrap \'kit_appdata_teardown\' EXIT\npython -u ' + shlex.quote(str(directory / 'run.py')) + '\nKIT_RC=0\n'
            (directory / 'job.sh').write_text(shell); (directory / 'pinned.sh').write_text('#!/bin/bash\nset -euo pipefail\nexec srun --cpu-bind=cores --exact -n1 -c8 bash ' + shlex.quote(str(directory / 'job.sh')) + '\n')
            subprocess.run(['bash', '-n', str(directory / 'job.sh')], check=True)
            tickets.append({'id': ident, 'physical_dedup_key': key, 'dedup_key': key, 'expected_account': 'minyeong',
                'claim_path': str(namespace / 'claims' / (key + '.json')), 'submission_path': str(namespace / 'receipts' / (key + '.json')),
                'sbatch_argv': ['sbatch', '--parsable', '--nodes=1', '--ntasks=1', '-A', 'vision', '-p', 'svl', '--exclude=svl9', '--gres=gpu:titanrtx:1', '-c', '8', '--mem=48G', '--time=02:00:00', '--nice=1000', '-J', 'isg261110_required_push_map', '-o', str(directory / 'slurm_%j.log'), str(directory / 'pinned.sh')],
                'prepared': ref(pp), 'runtime_ref': ref(runtime / '.snapshot_commit'), 'input_refs': inputs,
                'scripts': [ref(directory / n) for n in ('run.py', 'job.sh', 'pinned.sh')], 'output': str(directory / 'maps'),
                'dispatch_workdir': str(runtime), 'hard_time_seconds': 7200, 'native_time_seconds': 6900, 'computation_kind': 'contact_map',
                'consumer_kind': 'push_blade', 'workstream': 'training preparation', 'owner': '/root/templates_assets', 'stage': 'required axial push maps',
                'class_ids': classes, 'configuration_ids': configs, 'selected_dependency_ids': sorted(dep['id'] for dep, _ in selected),
                'benchmark_credit': 0, 'production_authorized': False, 'followup': 'Strict native ContactMap identity/coverage/grid/consumer lookup; zero is not positive.'})
    manifest = output / 'tickets.json'; write(manifest, {'schema': 'isg-preparation-dispatch-tickets-v1', 'campaign_id': root.name,
        'account': 'minyeong', 'source_account': 'minyeong', 'claim_namespace': str(namespace / 'claims'), 'receipt_namespace': str(namespace / 'receipts'),
        'observed_at': time.time(), 'owner': '/root/templates_assets', 'dispatch_owner': '/root', 'jobs': tickets,
        'retained': retained, 'missing_logical_requirements': dict(missing), 'production_authorized': False, 'benchmark_credit': 0})
    print(json.dumps({'manifest': ref(manifest), 'physical_jobs': len(tickets), 'retained': len(retained), 'missing_logical': dict(missing)}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'runtime', 'audit', 'cache', 'graph', 'output'): p.add_argument('--' + name, required=True)
    a = p.parse_args(); prepare(a.root, a.runtime, a.audit, a.cache, a.graph, a.output)
