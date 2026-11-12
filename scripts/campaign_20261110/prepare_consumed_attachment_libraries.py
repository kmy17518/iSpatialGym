"""Prepare finite attachment-free library builders, deduplicated per male/scale."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


def main(a):
    root, runtime, inventory, output = map(Path, (a.root, a.runtime, a.inventory, a.output))
    sys.path[:0] = [str(runtime / 'Behavior-1K/bddl3'), str(runtime)]
    from ispatialgym.grasping import schema as S
    from ispatialgym.sampling.knowledge import is_ready
    from scripts.campaign_20261110.reservation_policy import active_policy
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    ref = lambda p: {'path': str(p), 'sha256': sha(p)}
    read = lambda p: json.loads(Path(p).read_text())
    wanted_path = inventory / 'wanted.json'
    wanted = read(wanted_path)
    primary = read(inventory / 'cache_inventory.json')
    secondary_path = Path(a.secondary_cache)
    secondary = read(secondary_path)
    claims_path = inventory / 'existing_claim_audit.json'
    claims = read(claims_path)
    assert not primary['errors'] and not secondary['errors'] and not claims['matches']
    found = {x['entry'] for doc in (primary, secondary) for x in doc['hits']}
    pointer, reservations, assets = active_policy(root)
    assert pointer and not output.exists()
    output.mkdir(parents=True)
    manifest = []
    source_refs = [ref(wanted_path), ref(inventory / 'cache_inventory.json'), ref(secondary_path), ref(claims_path), ref(runtime / '.snapshot_commit')]
    for filename in ('pipeline.py', 'candidates.py', 'validate_tabletop.py', 'schema.py'):
        source_refs.append(ref(runtime / 'ispatialgym/grasping' / filename))
    runner = '''import hashlib,json,os,subprocess,sys,time
from pathlib import Path
D=Path(__file__).parent;p=json.loads((D/'prepared.json').read_text());h=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
for r in p['input_refs']:assert h(r['path'])==r['sha256'],r['path']
argv=list(p['native_argv']);argv[0]=sys.executable
v={'started_at':time.time(),'job_id':os.environ['SLURM_JOB_ID'],'prepared_sha256':h(D/'prepared.json'),'physical_dedup_key':p['physical_dedup_key'],'native_runtime':p['native_runtime'],'argv':argv,'benchmark_credit':0}
(D/'started.json').write_text(json.dumps(v,indent=2)+'\\n')
with(D/'native.log').open('w')as log:
 try:rc=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=3300).returncode
 except subprocess.TimeoutExpired:rc=124
lib=D/'glib/models'/p['entry_id']/'library.json';done='GB_VALIDATE_DONE'in(D/'native.log').read_text(errors='replace')
v.update(finished_at=time.time(),returncode=rc,native_validator_done=done,library={'path':str(lib),'sha256':h(lib)}if lib.exists()else None,strict_native_reader_required=True,production_authorized=False)
(D/'terminal.json').write_text(json.dumps(v,indent=2)+'\\n');sys.exit(0 if rc==0 and done and lib.exists()else 2)
'''
    compile(runner, 'runner', 'exec')
    for entry, row in sorted(wanted.items()):
        if entry in found:
            continue
        t = row['target']
        assert t['ready'] and is_ready(t['model']) and t['model_key'] not in reservations['train_exclude_models']
        assert t['category'] not in reservations['train_exclude_categories'] and t['asset_sha256'] not in assets['reserved_asset_hashes']
        scale = list(S.canonical_scale(t['applied_scale']))
        selected_job = row.get('job')
        constraint = selected_job['constraint'] if selected_job else 'attachment_free'
        assert constraint in {'attachment_free', 'tabletop_insert', 'flower_stem_top'}
        rest_hint = selected_job.get('rest_hint_wxyz') if selected_job else None
        assert entry == S.entry_id(t['category'], t['model'], scale, constraint, rest_hint)
        asset = runtime / 'Behavior-1K/datasets' / t['dataset'] / 'objects' / t['category'] / t['model'] / 'usd' / (t['model'] + '.encrypted.usd')
        assert sha(asset) == t['asset_sha256']
        job = selected_job or {'id': entry, 'category': t['category'], 'model': t['model'], 'dataset': t['dataset'], 'scale': scale,
               'constraint': 'attachment_free', 'kind': 'attachment', 'rest_hint_wxyz': None, 'rest_pose': 'canonical',
               'legacy_key': None, 'source': 'consumed_attachment_male_scale', 'asset_sha256': t['asset_sha256']}
        physical = {'kind': 'grasp_library', 'job': job, 'generator_files': source_refs[5:], 'arm': 'right'}
        key = hashlib.sha256(json.dumps(physical, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        assert all(job[k] == t[k] for k in ('category', 'model', 'dataset')) and job['scale'] == scale
        claims_root = root / 'infra/coordination/preparation_dispatch/consumed_asset_builds_r1'
        if (claims_root / 'claims' / (key + '.json')).exists():
            continue
        ident = ('attachment_glib_' if constraint == 'attachment_free' else 'specialized_glib_') + key[:20]
        directory = output / ident
        directory.mkdir()
        jp = directory / 'jobs.json'
        jp.write_text(json.dumps([job], indent=2) + '\n')
        argv = ['python', '-u', '-m', 'ispatialgym.grasping.pipeline', 'all', '--jobs', str(jp), '--root', str(directory / 'glib'), '--shard', '0/1', '--workers', '8']
        inputs = source_refs + row.get('source_refs', []) + [ref(jp), ref(asset)]
        for reference in inputs: assert sha(reference['path']) == reference['sha256']
        prepared = {'id': ident, 'physical_dedup_key': key, 'physical_identity': physical, 'entry_id': entry,
                    'native_runtime': str(runtime), 'native_argv': argv, 'input_refs': inputs,
                    'configuration_ids': sorted(set(row['consumers'])), 'selected_dependency_ids': sorted(row['request_ids']),
                    'computation_kind': 'grasp_library', 'consumer_kind': constraint, 'production_authorized': False}
        pp = directory / 'prepared.json'
        pp.write_text(json.dumps(prepared, indent=2) + '\n')
        (directory / 'run.py').write_text(runner)
        env = {'ISG_GB_ROOT': str(runtime), 'GBH': str(root), 'KIT_TPL_NFS_ROOT': str(root / 'kit_templates'),
               'KIT_TEX_STORE': str(root / 'kit_textures'), 'EXP_PATH': str(root / 'isaacsim_overlay/apps')}
        shell = '#!/bin/bash\nset -euo pipefail\n' + '\n'.join('export ' + k + '=' + shlex.quote(v) for k, v in env.items())
        shell += '\nset +u\nsource "$ISG_GB_ROOT/scripts/grasp_bands/env_cluster.sh"\nset -u\nexport PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR="$JB/numba_cache"\nKIT_RC=1\ntrap \'kit_appdata_teardown\' EXIT\npython -u ' + shlex.quote(str(directory / 'run.py')) + '\nKIT_RC=0\n'
        (directory / 'job.sh').write_text(shell)
        (directory / 'pinned.sh').write_text('#!/bin/bash\nset -euo pipefail\nexec srun --cpu-bind=cores --exact -n1 -c8 bash ' + shlex.quote(str(directory / 'job.sh')) + '\n')
        subprocess.run(['bash', '-n', str(directory / 'job.sh')], check=True)
        claims_root = root / 'infra/coordination/preparation_dispatch/consumed_asset_builds_r1'
        manifest.append({'id': ident, 'physical_dedup_key': key, 'dedup_key': key, 'expected_account': 'minyeong',
                         'claim_path': str(claims_root / 'claims' / (key + '.json')), 'submission_path': str(claims_root / 'receipts' / (key + '.json')),
                         'sbatch_argv': ['sbatch', '--parsable', '--nodes=1', '--ntasks=1', '-A', 'vision', '-p', 'svl,viscam', '--exclude=svl9', '--gres=gpu:titanrtx:1', '-c', '8', '--mem=48G', '--time=01:00:00', '--nice=1000', '-J', 'isg261110_attachment_glib', '-o', str(directory / 'slurm_%j.log'), str(directory / 'pinned.sh')],
                         'prepared': ref(pp), 'runtime_ref': ref(runtime / '.snapshot_commit'), 'input_refs': inputs,
                         'scripts': [ref(directory / n) for n in ('run.py', 'job.sh', 'pinned.sh')], 'output': str(directory / 'glib'),
                         'dispatch_workdir': str(runtime), 'hard_time_seconds': 3600, 'native_time_seconds': 3300,
                         'computation_kind': 'grasp_library', 'consumer_kind': constraint, 'workstream': 'training preparation',
                         'stage': 'native ' + constraint + ' grasp library', 'owner': '/root/templates_assets',
                         'configuration_ids': prepared['configuration_ids'], 'selected_dependency_ids': row['request_ids'],
                         'followup': 'Strict native entry/scale/constraint/rest/arm/physics/candidate-default and positive-versus-zero reader; link exact specialized consumer aliases and feed required maps through existing exact claims. No scene feasibility test.',
                         'benchmark_credit': 0, 'production_authorized': False})
    p = output / 'tickets.json'
    p.write_text(json.dumps({'schema': 'isg-preparation-dispatch-tickets-v1', 'campaign_id': root.name, 'account': 'minyeong',
                            'source_account': 'minyeong', 'claim_namespace': str(root / 'infra/coordination/preparation_dispatch/consumed_asset_builds_r1/claims'),
                            'receipt_namespace': str(root / 'infra/coordination/preparation_dispatch/consumed_asset_builds_r1/receipts'),
                            'observed_at': time.time(), 'owner': '/root/templates_assets', 'dispatch_owner': '/root', 'jobs': manifest,
                            'skipped_existing_entries': sorted(found & set(wanted)), 'reservation_policy': pointer,
                            'benchmark_credit': 0, 'production_authorized': False}, indent=2) + '\n')
    print(json.dumps({'manifest': ref(p), 'jobs': len(manifest), 'skipped_existing': len(found & set(wanted))}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'runtime', 'inventory', 'output', 'secondary-cache'):
        p.add_argument('--' + name, required=True)
    main(p.parse_args())
