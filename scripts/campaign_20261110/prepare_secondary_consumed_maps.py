"""Own-output secondary routes for already-readable exact map input closures."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


def prepare(source_manifest, allowed_keys, output):
    source_manifest, allowed_keys, output = map(Path, (source_manifest, allowed_keys, output))
    primary = Path('/vision/u/minyeong/projects/isg_home/campaign_20261110')
    secondary = Path('/vision/u/shiyuc/minyeong/campaign_20261110')
    assert output.is_relative_to(secondary) and not output.exists()
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    ref = lambda p: {'path': str(p), 'sha256': sha(p)}
    allowed = set(json.loads(allowed_keys.read_text())['physical_dedup_keys'])
    source = json.loads(source_manifest.read_text()); output.mkdir(parents=True)
    rows = []; skipped = []; errors = []
    for original in source['jobs']:
        key = original['physical_dedup_key']
        if key not in allowed: continue
        if Path(original['claim_path']).exists() or Path(original['submission_path']).exists():
            skipped.append({'physical_dedup_key': key, 'reason': 'existing physical ownership'}); continue
        try:
            assert original['computation_kind'] == 'reach_map' and original['consumer_kind'] == 'tabletop'
            for r in original['input_refs'] + original['scripts'] + [original['prepared'], original['runtime_ref']]:
                assert sha(r['path']) == r['sha256'], r['path']
            prepared = json.loads(Path(original['prepared']['path']).read_text()); target = output / original['id']; target.mkdir()
            native = Path(prepared['native_runtime']); assert native.is_relative_to(primary / 'code')
            for arm, argv in prepared['native_argv_by_arm'].items(): argv[argv.index('--out') + 1] = str(target / 'maps')
            prepared['source_original_prepared'] = original['prepared']; prepared['route_account'] = 'shiyuc'
            pp = target / 'prepared.json'; pp.write_text(json.dumps(prepared, indent=2) + '\n')
            # These are readable execution instructions; no library or geometry
            # payload is copied, and the source inputs keep their exact paths.
            run = next(r for r in original['scripts'] if Path(r['path']).name == 'run.py')
            (target / 'run.py').write_text(Path(run['path']).read_text())
            env = {'ISG_GB_ROOT': str(native), 'GBH': str(secondary), 'KIT_TPL_NFS_ROOT': str(secondary / 'kit_templates'),
                   'KIT_TEX_STORE': str(secondary / 'kit_textures'), 'EXP_PATH': str(secondary / 'isaacsim_overlay/apps')}
            shell = '#!/bin/bash\nset -euo pipefail\n' + '\n'.join('export ' + k + '=' + shlex.quote(v) for k, v in env.items())
            shell += '\nset +u\nsource "$ISG_GB_ROOT/scripts/grasp_bands/env_cluster.sh"\nset -u\nexport PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR="$JB/numba_cache"\nKIT_RC=1\ntrap \'kit_appdata_teardown\' EXIT\npython -u ' + shlex.quote(str(target / 'run.py')) + '\nKIT_RC=0\n'
            (target / 'job.sh').write_text(shell); (target / 'pinned.sh').write_text('#!/bin/bash\nset -euo pipefail\nexec srun --cpu-bind=cores --exact -n1 -c8 bash ' + shlex.quote(str(target / 'job.sh')) + '\n')
            subprocess.run(['bash', '-n', str(target / 'job.sh')], check=True)
            row = copy.deepcopy(original); row.update(expected_account='shiyuc', source_account='shiyuc', prepared=ref(pp),
                scripts=[ref(target / n) for n in ('run.py', 'job.sh', 'pinned.sh')], output=str(target / 'maps'),
                source_primary_ticket={'manifest': ref(source_manifest), 'physical_dedup_key': key}, source_access_allowlist=ref(allowed_keys))
            row['sbatch_argv'][-1] = str(target / 'pinned.sh'); row['sbatch_argv'][row['sbatch_argv'].index('-o') + 1] = str(target / 'slurm_%j.log')
            assert row['physical_dedup_key'] == row['dedup_key'] == prepared['physical_dedup_key']
            rows.append(row)
        except (OSError, ValueError, KeyError, AssertionError) as exc:
            errors.append({'physical_dedup_key': key, 'blocker': type(exc).__name__ + ': ' + str(exc)})
    manifest = output / 'tickets.json'; manifest.write_text(json.dumps({'schema': 'isg-preparation-dispatch-tickets-v1', 'campaign_id': primary.name,
        'account': 'shiyuc', 'source_account': 'shiyuc', 'observed_at': time.time(), 'source_manifest': ref(source_manifest), 'source_access_allowlist': ref(allowed_keys),
        'claim_namespace': str(primary / 'infra/coordination/preparation_dispatch/consumed_asset_builds_r1/claims'),
        'receipt_namespace': str(primary / 'infra/coordination/preparation_dispatch/consumed_asset_builds_r1/receipts'),
        'jobs': rows, 'skipped': skipped, 'errors': errors, 'owner': '/root/templates_assets', 'dispatch_owner': '/root',
        'benchmark_credit': 0, 'production_authorized': False}, indent=2) + '\n')
    print(json.dumps({'manifest': ref(manifest), 'ready': len(rows), 'claimed_skipped': len(skipped), 'errors': len(errors)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-manifest', 'allowed-keys', 'output'): parser.add_argument('--' + name, required=True)
    a = parser.parse_args(); prepare(a.source_manifest, a.allowed_keys, a.output)
