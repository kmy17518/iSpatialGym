"""Typed, immutable library preparation on the existing private Vast worker.

Only native dependency generation is supported. No scene sampler, collection,
acceptance, or arbitrary command can enter this route.
"""
import argparse
import copy
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


def consumed_attachment_ticket(root, tickets, index, native):
    """Read an approved physical library recipe without broadening its command."""
    source = json.loads(tickets.read_text())
    require(source['schema'] == 'isg-preparation-dispatch-tickets-v1', 'unsupported consumed library tickets')
    require(type(index) is int and 0 <= index < len(source['jobs']), 'invalid source entry index')
    ticket = source['jobs'][index]
    require(ticket['computation_kind'] == 'grasp_library' and ticket['consumer_kind'] == 'attachment_free', 'unsupported consumed library kind')
    key = ticket['physical_dedup_key']
    require(re.fullmatch(r'[a-f0-9]{64}', key) and key == ticket['dedup_key'], 'physical library key differs')
    for field in ('claim_path', 'submission_path'):
        path = Path(ticket[field])
        require(path.is_relative_to(root / 'infra/coordination/preparation_dispatch') and path.name == key + '.json', 'physical claim ownership differs')
    require(ticket['native_time_seconds'] == 3300 and ticket['hard_time_seconds'] == 3600, 'attachment library bounds differ')
    require(ticket['benchmark_credit'] == 0 and ticket['production_authorized'] is False, 'library purpose differs')
    refs = ticket['input_refs'] + ticket['scripts'] + [ticket['prepared'], ticket['runtime_ref']]
    files = {}
    for ref in refs:
        path = Path(ref['path'])
        require(path.is_relative_to(root) and sha(path) == ref['sha256'], 'consumed library input changed: ' + str(path))
        require(str(path) not in files or files[str(path)] == ref['sha256'], 'conflicting library input hash')
        files[str(path)] = ref['sha256']
    require(ticket['runtime_ref']['path'] == str(native / '.snapshot_commit'), 'consumed library runtime differs')
    prepared = json.loads(Path(ticket['prepared']['path']).read_text())
    require(prepared['physical_dedup_key'] == key and prepared['native_runtime'] == str(native), 'prepared library identity differs')
    argv = prepared['native_argv']
    require(argv[:6] == ['python', '-u', '-m', 'ispatialgym.grasping.pipeline', 'all', '--jobs'] and
            len(argv) == 13 and argv[7] == '--root' and argv[9:] == ['--shard', '0/1', '--workers', '8'], 'consumed library command differs')
    manifest = Path(argv[6])
    require(str(manifest) in files, 'native library manifest is not pinned')
    rows = json.loads(manifest.read_text())
    require(len(rows) == 1, 'consumed library must contain one physical entry')
    entry = rows[0]
    require(entry['kind'] == 'attachment' and entry['constraint'] == 'attachment_free' and
            entry['rest_pose'] == 'canonical' and entry['rest_hint_wxyz'] is None and entry['legacy_key'] is None,
            'unsupported attachment native entry')
    require(entry['id'] == prepared['entry_id'] and prepared['physical_identity']['job'] == entry, 'attachment entry differs')
    identity_key = hashlib.sha256(json.dumps(prepared['physical_identity'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(identity_key == key and prepared['physical_identity']['arm'] == 'right', 'physical library fingerprint differs')
    asset = native / 'Behavior-1K/datasets' / entry['dataset'] / 'objects' / entry['category'] / entry['model'] / 'usd' / (entry['model'] + '.encrypted.usd')
    require(files.get(str(asset)) == entry['asset_sha256'], 'attachment asset closure differs')
    require(ticket['configuration_ids'] == prepared['configuration_ids'] and bool(ticket['configuration_ids']), 'selected library consumers differ')
    return ticket, manifest, entry, asset, files


def prepare_consumed_attachment(root, tickets, index, native, helper, token):
    """Transport an exact attachment-free builder; shared claim remains external."""
    root, tickets, native, helper = map(lambda p: Path(p).resolve(), (root, tickets, native, helper))
    require(re.fullmatch(r'[A-Za-z0-9_-]+', token), 'unsafe prebuild token')
    require(tickets.is_relative_to(root / 'training_preparation'), 'prebuild request outside training preparation')
    require(native.is_relative_to(root / 'code') and helper.is_relative_to(root / 'code'), 'frozen runtimes required')
    ticket, manifest, entry, asset, files = consumed_attachment_ticket(root, tickets, index, native)
    job = root / 'out/vast_prebuild' / token
    require(not job.exists(), 'prebuild token already has evidence')
    job.mkdir(parents=True)
    selected = job / 'selected_jobs.json'; write(selected, [entry])
    env = {'ISGE_CODE': str(native), 'ISGF_ROOT': str(root), 'ISGF_NATIVE_HELPER': str(helper)}
    (job / 'job.env').write_text(''.join(k + '=' + shlex.quote(v) + '\n' for k, v in env.items()))
    for path in (tickets, selected, job / 'job.env', helper / 'scripts/campaign_20261110/vast_prebuild.py', helper / 'scripts/campaign_20261110/job_prebuild.sh'):
        files[str(path)] = sha(path)
    value = {'schema': SCHEMA, 'campaign_id': root.name, 'campaign_root': str(root),
             'job_dir': str(job), 'snapshot': str(native), 'helper_snapshot': str(helper),
             'source_manifest': str(manifest), 'source_request': str(tickets), 'entry_index': index,
             'original_entry': {**entry, 'class_id': 'dependency-attachment', 'task': 'library_preparation'},
             'selected_manifest': str(selected), 'files': files, 'asset_files': {str(asset): sha(asset)},
             'scenes': [], 'kind': 'attachment', 'environment': {'GBH': str(root), 'ISG_GB_ROOT': str(native)},
             'time_limit_s': 3300, 'allocation_time_s': 3600, 'prepared_at_unix': time.time(),
             'physical_dedup_key': ticket['physical_dedup_key'], 'claim_path': ticket['claim_path'],
             'submission_path': ticket['submission_path'], 'configuration_ids': ticket['configuration_ids'],
             'new_native_instances': 0, 'benchmark_credit': 0, 'production_released': False}
    write(job / 'native_inputs.json', value)
    (job / 'sample.log').symlink_to('result/native.log')
    return job


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
    require(sha(proof['path']) == proof['sha256'], 'native default measurement differs')
    job = root / 'out/vast_prebuild' / token
    require(not job.exists(), 'prebuild token already has evidence')
    job.mkdir(parents=True)
    # Private proof relocation changes no native scale/rest/model/geometry.
    private_proof = job / 'native_default_evidence.json'
    private_proof.write_bytes(Path(proof['path']).read_bytes())
    require(sha(private_proof) == proof['sha256'], 'private default measurement copy differs')
    selected_entry = copy.deepcopy(entry)
    selected_entry['native_default_evidence']['path'] = str(private_proof)
    selected = job / 'selected_jobs.json'
    write(selected, [selected_entry])
    env = {'ISGE_CODE': str(native), 'ISGF_ROOT': str(root), 'ISGF_NATIVE_HELPER': str(helper)}
    (job / 'job.env').write_text(''.join(k+'='+shlex.quote(v)+'\n' for k, v in env.items()))
    files = [request, manifest, private_proof, native / '.snapshot_commit',
             native / 'scripts/grasp_bands/job_glib.sh', selected, job / 'job.env',
             helper / 'scripts/campaign_20261110/vast_prebuild.py',
             helper / 'scripts/campaign_20261110/job_prebuild.sh']
    value = {'schema': SCHEMA, 'campaign_id': root.name, 'campaign_root': str(root),
             'job_dir': str(job), 'snapshot': str(native), 'helper_snapshot': str(helper),
             'source_manifest': str(manifest), 'source_request': str(request), 'entry_index': index,
             'original_entry': {**entry, 'class_id': 'dependency-tabletop', 'task': 'library_preparation'},
             'selected_manifest': str(selected), 'files': {str(p): sha(p) for p in files},
             'proof_relocation': {'original': proof['path'], 'copy': str(private_proof), 'sha256': proof['sha256']},
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
    require(value['kind'] in {'tabletop', 'attachment'} and value['new_native_instances'] == 0 and value['benchmark_credit'] == 0 and value['production_released'] is False, 'prebuild purpose differs')
    for name, expected in value['files'].items():
        path = Path(name)
        asset_link = name in value['asset_files'] and path.is_relative_to(Path(value['snapshot']) / 'Behavior-1K/datasets')
        require((path.resolve().is_relative_to(root) or asset_link) and sha(path) == expected, 'prebuild input changed: '+name)
    for name, expected in value['asset_files'].items():
        require(Path(name).is_relative_to(Path(value['snapshot']) / 'Behavior-1K/datasets') and sha(name) == expected, 'prebuild native asset changed: '+name)
    if value['kind'] == 'attachment':
        ticket, manifest, original, _, _ = consumed_attachment_ticket(root, Path(value['source_request']), value['entry_index'], Path(value['snapshot']))
        require(str(manifest) == value['source_manifest'], 'attachment source manifest differs')
        require(value['time_limit_s'] == 3300 and value['allocation_time_s'] == 3600, 'attachment library bounds differ')
        for field in ('physical_dedup_key', 'claim_path', 'submission_path', 'configuration_ids'):
            require(value[field] == ticket[field], 'attachment physical ownership differs')
        require(json.loads(Path(value['selected_manifest']).read_text()) == [original], 'prebuild selected native row differs')
        return value
    original = json.loads(Path(value['source_manifest']).read_text())[value['entry_index']]
    relocation = value['proof_relocation']
    require(original['native_default_evidence'] == {'path': relocation['original'], 'sha256': relocation['sha256']}, 'native measurement provenance differs')
    original['native_default_evidence']['path'] = relocation['copy']
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
              'job_dir': str(job), 'time_s': value.get('allocation_time_s', 9600), 'cpus': 9, 'mem': '48G', 'acct': 'vision', 'part': 'svl',
              'log': str(root / 'slurm' / ('prebuild_'+str(jid)+'.log')), 'cwd': str(root),
              'submitted': time.time(), 'submitted_by': 'typed_library_prebuild', 'gpu': gpu,
              'native_inputs_sha256': sha(job / 'native_inputs.json')}
    if value['kind'] == 'attachment':
        record.update({k: value[k] for k in ('physical_dedup_key', 'claim_path', 'submission_path')})
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
