"""Typed, immutable library preparation on the existing private Vast worker.

Only native dependency generation is supported. No scene sampler, collection,
acceptance, or arbitrary command can enter this route.
"""
import argparse
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

SCHEMA = 'isg-vast-library-prebuild-v1'


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n'); temporary.replace(path)


def library_environment(root, v2_root=None):
    return {}  # These jobs create libraries; they do not consume baseline maps.


def prepare(root, request, kind, index, native, helper, token):
    root, request, native, helper = map(lambda p: Path(p).resolve(), (root, request, native, helper))
    require(re.fullmatch(r'[A-Za-z0-9_-]+', token), 'unsafe prebuild token')
    require(request.is_relative_to(root / 'training_preparation'), 'prebuild request outside training preparation')
    require(native.is_relative_to(root / 'code') and helper.is_relative_to(root / 'code'), 'frozen runtimes required')
    require(kind == 'tabletop', 'floating requires a separately verified GraspGen closure')
    source = json.loads(request.read_text())
    batch = next(b for b in source['batches'] if b['kind'] == kind)
    require(batch['stages'] == 'glib', 'only exact native library generation is enabled')
    manifest = Path(batch['jobs'])
    require(manifest.resolve().is_relative_to(root), 'prebuild manifest outside campaign')
    require(sha(manifest) == batch['jobs_sha256'], 'prebuild manifest changed')
    rows = json.loads(manifest.read_text())
    require(type(index) is int and 0 <= index < len(rows), 'invalid source entry index')
    entry = rows[index]
    require(entry['ready'] is True and entry['kind'] == 'tabletop' and entry['constraint'] == 'tabletop', 'unsupported native library entry')
    require(not any(c.startswith('configuration_matching-block_design') for c in entry.get('campaign_classes', [])), 'block-design selection remains under review')
    asset = Path(entry['asset_path'])
    require(asset.is_relative_to(native / 'Behavior-1K/datasets') and sha(asset) == entry['asset_sha256'], 'exact asset identity differs')
    proof = entry['native_default_evidence']
    require(Path(proof['path']).resolve().is_relative_to(root) and sha(proof['path']) == proof['sha256'], 'native default measurement differs')
    job = root / 'out/vast_prebuild' / token
    require(not job.exists(), 'prebuild token already has evidence')
    job.mkdir(parents=True)
    # The exact row is copied without changing scale, rest, model, or proof.
    selected = job / 'selected_jobs.json'
    write(selected, [entry])
    env = {'ISGE_CODE': str(native), 'ISGF_ROOT': str(root), 'ISGF_NATIVE_HELPER': str(helper)}
    (job / 'job.env').write_text(''.join(k+'='+shlex.quote(v)+'\n' for k, v in env.items()))
    files = [request, manifest, Path(proof['path']), native / '.snapshot_commit',
             native / 'scripts/grasp_bands/job_glib.sh', selected, job / 'job.env',
             helper / 'scripts/campaign_20261110/vast_prebuild.py',
             helper / 'scripts/campaign_20261110/job_prebuild.sh']
    value = {'schema': SCHEMA, 'campaign_id': root.name, 'campaign_root': str(root),
             'job_dir': str(job), 'snapshot': str(native), 'helper_snapshot': str(helper),
             'source_manifest': str(manifest), 'source_request': str(request), 'entry_index': index,
             'original_entry': {**entry, 'class_id': 'dependency-tabletop', 'task': 'library_preparation'},
             'selected_manifest': str(selected), 'files': {str(p): sha(p) for p in files},
             'asset_files': {str(asset): sha(asset)}, 'scenes': [], 'kind': kind,
             'environment': {'GBH': str(root), 'ISG_GB_ROOT': str(native)},
             'time_limit_s': 9000, 'prepared_at_unix': time.time(),
             'new_native_instances': 0, 'benchmark_credit': 0, 'production_released': False}
    write(job / 'native_inputs.json', value)
    (job / 'sample.log').symlink_to('result/native.log')
    return job


def verify(job, root=None):
    job = Path(job).resolve()
    value = json.loads((job / 'native_inputs.json').read_text())
    root = Path(root or value['campaign_root']).resolve()
    require(value['schema'] == SCHEMA and value['campaign_root'] == str(root), 'prebuild campaign differs')
    require(job.is_relative_to(root / 'out/vast_prebuild') and value['job_dir'] == str(job), 'prebuild output ownership differs')
    require(value['kind'] == 'tabletop' and value['new_native_instances'] == 0 and value['production_released'] is False, 'prebuild purpose differs')
    for name, expected in value['files'].items():
        path = Path(name)
        require(path.resolve().is_relative_to(root) and sha(path) == expected, 'prebuild input changed: '+name)
    for name, expected in value['asset_files'].items():
        require(Path(name).is_relative_to(Path(value['snapshot']) / 'Behavior-1K/datasets') and sha(name) == expected, 'prebuild native asset changed: '+name)
    original = json.loads(Path(value['source_manifest']).read_text())[value['entry_index']]
    selected = json.loads(Path(value['selected_manifest']).read_text())
    require(selected == [original], 'prebuild selected native row differs')
    return value


def enqueue(job, root, gpu='RTX 4090'):
    root, job = Path(root).resolve(), Path(job).resolve()
    value = verify(job, root)
    receipt = job / 'vast_submission.json'
    if receipt.exists():
        old = json.loads(receipt.read_text())
        require(sha(root / 'vast/queue' / (str(old['id'])+'.json')) == old['record_sha256'], 'prebuild queue changed')
        return str(old['id'])
    ids = root / 'vast/queue/ids'; ids.mkdir(parents=True, exist_ok=True)
    jid = max([900000000]+[int(p.name) for p in ids.iterdir() if p.name.isdigit()])+1
    while True:
        try: (ids / str(jid)).mkdir(); break
        except FileExistsError: jid += 1
    record = {'id': jid, 'job_kind': 'native_dependency', 'name': 'isg261110_library_prebuild',
              'script': str(Path(value['helper_snapshot']) / 'scripts/campaign_20261110/job_prebuild.sh'),
              'job_dir': str(job), 'time_s': 9600, 'cpus': 9, 'mem': '48G', 'acct': 'vision', 'part': 'svl',
              'log': str(root / 'slurm' / ('prebuild_'+str(jid)+'.log')), 'cwd': str(root),
              'submitted': time.time(), 'submitted_by': 'typed_library_prebuild', 'gpu': gpu,
              'native_inputs_sha256': sha(job / 'native_inputs.json')}
    target = root / 'vast/queue' / (str(jid)+'.json'); write(target, record)
    write(receipt, {'id': jid, 'record_sha256': sha(target), 'activated': False})
    return str(jid)


def run(job):
    job = Path(job).resolve(); value = verify(job)
    output = job / 'result'; output.mkdir(exist_ok=False)
    argv = [sys.executable, '-u', '-m', 'ispatialgym.grasping.pipeline', 'all', '--jobs',
            value['selected_manifest'], '--root', str(output / 'glib'), '--shard', '0/1',
            '--workers', os.environ.get('SLURM_CPUS_PER_TASK', '8')]
    started = time.time()
    write(output / 'command.json', {'argv': argv, 'started_at': started, 'snapshot': value['snapshot'],
                                  'source_manifest_sha256': value['files'][value['source_manifest']]})
    with (output / 'native.log').open('w') as log:
        try: rc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=value['time_limit_s']).returncode
        except subprocess.TimeoutExpired: rc = 124
    done = 'GB_VALIDATE_DONE' in (output / 'native.log').read_text(errors='replace')
    paths = sorted((output / 'glib').rglob('library.json'))
    report = {'returncode': rc, 'native_validation_done': done, 'library_files': {str(p): sha(p) for p in paths},
              'positive_native_reader_required': True, 'new_native_instances': 0, 'benchmark_credit': 0,
              'started_at': started, 'finished_at': time.time()}
    write(output / 'prebuild_result.json', report)
    return 0 if rc == 0 and done and paths else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--verify'); parser.add_argument('--run')
    args = parser.parse_args()
    if args.verify: verify(args.verify); print('exact native prebuild inputs verified')
    elif args.run: sys.exit(run(args.run))
    else: parser.error('choose --verify or --run')
