"""Prepare deduplicated wall-frame maps from strictly verified contact physics."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def main(args):
    root, base, out, runtime = map(Path, (args.root, args.account_root, args.output, args.runtime))
    sys.path[:0] = [str(runtime / 'Behavior-1K/bddl3'), str(runtime)]
    from ispatialgym.grasping import contact_library as CL, contact_policy as CP
    from ispatialgym.sampling import contact_maps as CM, reach_maps as RM
    cache = {}
    copies = json.loads(Path(args.source_copies).read_text()) if args.source_copies else {}
    def resolve(p):
        if str(p) in copies:
            copy = copies[str(p)]
            dest = Path(copy['path'])
            assert hashlib.sha256(dest.read_bytes()).hexdigest() == copy['sha256']
            return dest
        return Path(p)
    def sha(p):
        p = str(p)
        if p not in cache:
            cache[p] = hashlib.sha256(resolve(p).read_bytes()).hexdigest()
        return cache[p]
    def ref(p):
        return {'path': str(resolve(p)), 'sha256': sha(p)}
    def read(p):
        return json.loads(resolve(p).read_text())
    def write(p, d):
        p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with p.open('x') as f:
            json.dump(d, f, indent=2)
            f.write('\n')
    assert not out.exists()
    out.mkdir(parents=True, mode=0o700)
    os.umask(0o077)
    inventory = root / 'training_preparation/model_variant_inventory_r2'
    graph_path = inventory / 'broad_map_inputs_r1/selected_manifest.json'
    graph = read(graph_path)
    deps = [r for r in graph['artifact_requests'] if r['kind'] == 'contact_brace_map' and any(c.startswith('alignment-world_vertical_') for c in r['required_by_classes'])]
    canonical_path = root / 'infra/standard_recovery/canonical_validation.json'
    canonical_bytes = resolve(canonical_path).read_bytes()
    snapshot = out / 'canonical_source.json'
    snapshot.write_bytes(canonical_bytes)
    canonical = {c['class_id']: c for c in json.loads(canonical_bytes)['classes']}
    classes = {}
    for cid in {c for d in deps for c in d['required_by_classes']}:
        c = canonical[cid]
        directory = Path(c['configuration']['scene_setup']['source_template'])
        meta_path = directory / 'template_meta.json'
        meta = read(meta_path)
        custom_path = Path(meta['customs'])
        generation = read(custom_path)[meta.get('alias', meta.get('variant'))]['generation']
        assert generation['hang_height_m'] == 1.45 and generation['stance_arm_posture'] == 'tucked'
        classes[cid] = {'hinge_height_m': generation['hang_height_m'], 'customs': ref(custom_path), 'metadata': ref(meta_path)}
    existing = []
    if args.existing_map_audit:
        owner_audit = read(args.existing_map_audit)
        assert owner_audit['owner'] == '/root/templates_assets'
        existing = owner_audit['wall_units']
    else:
        for p in (root / 'training_preparation').glob('*contact*/request*.json'):
            d = read(p)
            for unit in d.get('units', []):
                if unit.get('map_kind') == 'wall_brace':
                    existing.append({'source': ref(p), 'unit': unit})
    # Earlier contact-map streams explicitly exclude wall-frame queries. If one
    # appears, reconcile its exact ownership before making a competing request.
    assert not existing, 'Existing wall-map owner requires exact reconciliation'
    robot = RM.robot_digest()
    marker = ref(runtime / '.snapshot_commit')
    jobs, skipped, seen = [], [], set()
    runner = '''from pathlib import Path
import hashlib,json,os,subprocess,sys,time
K=Path(__file__).parent;p=json.loads((K/'prepared.json').read_text());sha=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
for r in p['input_refs']:assert sha(r['path'])==r['sha256'],r['path']
argv=list(p['native_argv']);argv[0]=sys.executable
rec={'started_at':time.time(),'job_id':os.environ.get('SLURM_JOB_ID'),'physical_dedup_key':p['physical_dedup_key'],'prepared_sha256':sha(K/'prepared.json'),'argv':argv,'native_runtime':p['native_runtime'],'benchmark_credit':0}
(K/'started.json').write_text(json.dumps(rec,indent=2)+'\\n')
with(K/'native.log').open('w')as log:
 try:rc=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=6900).returncode
 except subprocess.TimeoutExpired:rc=124
maps=[{'path':str(f),'sha256':sha(f)}for f in(K/'maps').rglob('cmap__*')if f.suffix in('.json','.npz')]
rec.update(finished_at=time.time(),returncode=rc,maps=maps,native_map_done='GB_CONTACT_MAPS_DONE'in(K/'native.log').read_text(errors='replace'),strict_native_reader_required=True,production_authorized=False)
(K/'terminal.json').write_text(json.dumps(rec,indent=2)+'\\n')
sys.exit(0 if rc==0 and rec['native_map_done'] and maps else 2)
'''
    compile(runner, 'run.py', 'exec')
    for audit_path in map(Path, args.audit):
        audit = read(audit_path)
        for row in audit['rows']:
            if row['status'] != 'verified_native_contact_physics':
                skipped.append({'audit': ref(audit_path), 'source_job': row['source_job'], 'status': row['status']})
                continue
            for scope in row['map_scopes']:
                if scope['map_kind'] != 'wall_brace':
                    continue
                assert sha(row['library']['path']) == row['library']['sha256'] and sha(row['prepared']['path']) == row['prepared']['sha256']
                lib = read(row['library']['path'])
                prep = read(row['prepared']['path'])
                obj, arm = row['object'], scope['map_arm']
                assert lib['digest'] == CL._library_digest(lib)
                selected = [d for d in deps if d['inputs']['arm'] == arm and all(d['inputs']['target'][k] == obj[k] for k in ('category', 'model')) and d['inputs']['target']['applied_scale'] == obj['scale']]
                assert selected, (obj['entry'], arm, 'no selected dependency')
                config = CM.contact_config('wall_brace', arm=arm, lattice_m=.08, seeds=8).key()
                physical = {'kind': 'wall_brace', 'arm': arm, 'height_above_floor': 1.45, 'object_key': CM.object_key(obj), 'pose_set': scope['map_pose_set'], 'config': config, 'robot': robot, 'policy_key': CP.DEFAULT.key(arm)}
                key = hashlib.sha256(json.dumps(physical, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                job_id = 'wall_map_' + key[:20]
                directory = out / job_id
                directory.mkdir(mode=0o700)
                model_dir = directory / 'glib/models' / obj['entry']
                model_dir.mkdir(parents=True, mode=0o700)
                inputs = [ref(audit_path), row['library'], row['prepared'], row['terminal'], marker, ref(snapshot), ref(graph_path)]
                if args.existing_map_audit:
                    inputs.append(ref(args.existing_map_audit))
                for p, h in prep['private_model_files'].items():
                    assert sha(p) == h
                    dest = model_dir / Path(p).name
                    dest.write_bytes(Path(p).read_bytes())
                    assert sha(dest) == h
                    inputs.append(ref(dest))
                native = deepcopy(lib)
                native['object']['source'] = str(model_dir)
                assert CL._library_digest(native) == lib['digest']
                library = directory / 'contacts/brace/models' / obj['entry'] / 'library.json'
                write(library, native)
                inputs.append(ref(library))
                consumer_classes = sorted({c for d in selected for c in d['required_by_classes']})
                proofs = {c: classes[c] for c in consumer_classes}
                inputs.extend(r for p in proofs.values() for r in (p['customs'], p['metadata']))
                argv = ['python', '-u', '-m', 'ispatialgym.sampling.contact_maps', 'compute', '--root', str(directory / 'contacts'), '--kind', 'wall_brace', '--arm', arm, '--only', obj['entry'], '--heights', '1.4500', '--verdicts', 'valid', '--lattice', '0.08', '--seeds', '8', '--out', str(directory / 'maps')]
                prepared = {'physical_dedup_key': key, 'physical_identity': physical, 'native_runtime': str(runtime), 'source_audit': ref(audit_path), 'source_library': row['library'], 'native_library': ref(library), 'transport_only_change': 'object.source relocated to exact private geometry', 'object': obj, 'arm': arm, 'map_kind': 'wall_brace', 'height': 1.45, 'height_key': '1.4500', 'expected_map_config': config, 'pose_set': scope['map_pose_set'], 'input_refs': inputs, 'native_argv': argv, 'configuration_ids': sorted({c for d in selected for c in d['consumers']}), 'selected_dependency_ids': sorted(d['id'] for d in selected), 'class_ids': consumer_classes, 'height_proofs': proofs, 'benchmark_credit': 0, 'production_authorized': False}
                write(directory / 'prepared.json', prepared)
                (directory / 'run.py').write_text(runner)
                env = {'ISG_GB_ROOT': str(runtime), 'GBH': str(base), 'KIT_TPL_NFS_ROOT': str(base / 'kit_templates'), 'KIT_TEX_STORE': str(base / 'kit_textures'), 'EXP_PATH': str(base / 'isaacsim_overlay/apps')}
                shell = '#!/bin/bash\nset -euo pipefail\n' + '\n'.join('export ' + k + '=' + shlex.quote(v) for k, v in env.items()) + '\nset +u\nsource "$ISG_GB_ROOT/scripts/grasp_bands/env_cluster.sh"\nset -u\nexport PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR="$JB/numba_cache"\nKIT_RC=1\ntrap \'kit_appdata_teardown\' EXIT\npython -u ' + shlex.quote(str(directory / 'run.py')) + '\nKIT_RC=0\n'
                (directory / 'job.sh').write_text(shell)
                (directory / 'pinned.sh').write_text('#!/bin/bash\nset -euo pipefail\nexec srun --cpu-bind=cores --exact -n1 -c8 bash ' + shlex.quote(str(directory / 'job.sh')) + '\n')
                subprocess.run(['bash', '-n', str(directory / 'job.sh')], check=True)
                claims = inventory / 'wall_contact_maps_execution_r1'
                jobs.append({'id': job_id, 'physical_dedup_key': key, 'dedup_key': key, 'expected_account': args.account, 'claim_path': str(claims / 'claims' / (key + '.json')), 'submission_path': str(claims / 'receipts' / (key + '.json')), 'sbatch_argv': ['sbatch', '--parsable', '--nodes=1', '--ntasks=1', '-A', 'vision', '-p', 'svl,viscam', '--exclude=svl9', '--gres=gpu:titanrtx:1', '-c', '8', '--mem=48G', '--time=02:00:00', '--nice=1000', '-J', 'isg261110_wall_contact_map', '-o', str(directory / 'slurm_%j.log'), str(directory / 'pinned.sh')], 'prepared': ref(directory / 'prepared.json'), 'runtime_ref': marker, 'input_refs': inputs, 'scripts': [ref(directory / n) for n in ('run.py', 'job.sh', 'pinned.sh')], 'output': str(directory / 'maps'), 'dispatch_workdir': str(runtime), 'hard_time_seconds': 7200, 'native_time_seconds': 6900, 'workstream': 'training preparation', 'stage': 'native wall-frame contact map', 'owner': '/root/templates_assets', 'class_ids': consumer_classes, 'configuration_ids': prepared['configuration_ids'], 'selected_dependency_ids': prepared['selected_dependency_ids'], 'arm': arm, 'status': 'prepared_unsubmitted', 'followup': 'Strict ContactMap reader must verify native robot/policy/config/object/rest/pose-set/arm/height, complete trunk lattice and positive versus zero cells before graph join.', 'benchmark_credit': 0, 'production_authorized': False})
    write(out / 'tickets.json', {'schema': 'isg-preparation-dispatch-tickets-v1', 'campaign_id': root.name, 'account': args.account, 'jobs': jobs, 'skipped': skipped, 'existing_wall_units': existing, 'owner': '/root/templates_assets', 'dispatch_owner': '/root', 'observed_at': time.time(), 'benchmark_credit': 0, 'production_authorized': False})
    print(json.dumps({'manifest': ref(out / 'tickets.json'), 'jobs': len(jobs), 'arms': dict(Counter(j['arm'] for j in jobs))}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'account-root', 'output', 'runtime', 'account'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--audit', action='append', required=True)
    parser.add_argument('--source-copies')
    parser.add_argument('--existing-map-audit')
    main(parser.parse_args())
