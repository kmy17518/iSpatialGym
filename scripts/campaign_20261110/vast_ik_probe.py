"""Typed, permanently nonaccepting transport for a bounded native IK prefix."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.campaign_20261110 import vast_native_dependency as dependency
from scripts.campaign_20261110.ik_native_probe import validate_spec

SCHEMA = 'isg-vast-ik-native-probe-v1'
sha, write, require = dependency.sha, dependency.write, dependency.require
library_environment = dependency.library_environment


def prepare(root, spec_path, helper, environment, token, v2_root):
    root, spec_path, helper, release = map(lambda p: Path(p).resolve(),
                                         (root, spec_path, helper, v2_root))
    require(root.name == 'campaign_20261110' and re.fullmatch(r'[A-Za-z0-9_-]+', token), 'foreign probe')
    require(all(p.is_relative_to(root) for p in (spec_path, helper, release)), 'probe input outside campaign')
    require(helper.is_relative_to(root / 'code'), 'frozen helper required')
    source = json.loads(spec_path.read_text())
    output = root / 'ik_exchange/probes' / token
    paths, _ = validate_spec(source, output)
    require(environment.get('ISPATIALGYM_CAMPAIGN_CONFIG') == str(paths['config']), 'probe config differs')
    require(environment.get('ISPATIALGYM_EXACT_IK_MANIFEST') == str(paths['manifest']), 'probe manifest differs')
    require(all(k.startswith(('ISPATIALGYM_', 'ISG_')) for k in environment), 'unsupported environment')
    expected = library_environment(root, release)
    for key in ('ISPATIALGYM_GRASP_LIBRARY_V2', 'ISPATIALGYM_REACH_MAPS', 'ISPATIALGYM_REACH_MAP_ANCHORS'):
        require(environment.get(key) == expected[key], 'probe library routing differs: ' + key)
    job = root / 'out/vast_native_dependency' / token
    require(not job.exists(), 'probe token already has evidence')
    job.mkdir(parents=True)
    (job / 'sample.log').symlink_to('probe.log')
    files = dict(source['inputs'])
    files[str(spec_path)] = sha(spec_path)
    for name in ('vast_ik_probe.py', 'ik_native_probe.py', 'job_ik_native_probe.sh'):
        p = helper / 'scripts/campaign_20261110' / name
        files[str(p)] = sha(p)
    for p in paths['template'].glob('*.json'):
        files[str(p)] = sha(p)
    if source.get('bddl_root'):
        for p in Path(source['bddl_root']).rglob('*.bddl'):
            files[str(p)] = sha(p)
    manifest = release / 'release_manifest.json'
    library_files = {str(manifest): sha(manifest)}
    for name, digest in json.loads(manifest.read_text())['files'].items():
        p = release / name
        require(p.resolve().is_relative_to(release) and sha(p) == digest, 'release payload differs')
        library_files[str(p)] = digest
    env = {'ISGE_CODE': str(paths['snapshot']), 'ISGF_ROOT': str(root),
           'ISGF_NATIVE_HELPER': str(helper), **environment}
    (job / 'job.env').write_text(''.join(k + '=' + shlex.quote(v) + '\n' for k, v in env.items()))
    cfg = json.loads(paths['config'].read_text())
    value = {'schema': SCHEMA, 'campaign_id': root.name, 'campaign_root': str(root),
             'job_dir': str(job), 'snapshot': str(paths['snapshot']), 'helper_snapshot': str(helper),
             'probe_spec': str(spec_path), 'probe_output': str(output), 'environment': environment,
             'files': files, 'library_files': library_files, 'v2_release_root': str(release),
             'scenes': [cfg['template']['scene']], 'original_entry': {'class_id': cfg['template']['class_id']},
             'source_job': source.get('source_job'), 'fresh_start_index': None,
             'diagnostic_only': True, 'benchmark_credit': 0, 'prepared_at_unix': time.time()}
    write(job / 'native_inputs.json', value)
    return job


def verify(job, root=None):
    job = Path(job).resolve()
    value = json.loads((job / 'native_inputs.json').read_text())
    owned = Path(root or value['campaign_root']).resolve()
    require(value.get('schema') == SCHEMA and owned.name == 'campaign_20261110'
            and value.get('campaign_id') == owned.name and value['campaign_root'] == str(owned)
            and job.is_relative_to(owned / 'out/vast_native_dependency') and value['job_dir'] == str(job),
            'foreign probe ownership')
    require(value.get('diagnostic_only') is True and value.get('benchmark_credit') == 0,
            'probe cannot produce benchmark credit')
    for name, digest in {**value['files'], **value['library_files']}.items():
        p = Path(name)
        require(p.resolve().is_relative_to(owned) and p.is_file() and sha(p) == digest,
                'probe input bytes changed: ' + name)
    spec = json.loads(Path(value['probe_spec']).read_text())
    paths, output = validate_spec(spec, value['probe_output'])
    require(str(paths['snapshot']) == value['snapshot'], 'probe runtime differs')
    require(value['environment'].get('ISPATIALGYM_CAMPAIGN_CONFIG') == str(paths['config'])
            and value['environment'].get('ISPATIALGYM_EXACT_IK_MANIFEST') == str(paths['manifest']),
            'probe effective identity differs')
    return value


def enqueue(job, root, gpu='RTX 4090'):
    # The existing queue machinery preserves an immutable record and never
    # activates a job implicitly. Its verifier is selected by the typed schema.
    root, job = Path(root).resolve(), Path(job).resolve()
    value = verify(job, root)
    receipt = job / 'vast_submission.json'
    require(not receipt.exists(), 'probe already enqueued; reconcile its receipt')
    ids = root / 'vast/queue/ids'; ids.mkdir(parents=True, exist_ok=True)
    jid = max([900000000] + [int(p.name) for p in ids.iterdir() if p.name.isdigit()]) + 1
    while True:
        try: (ids / str(jid)).mkdir(); break
        except FileExistsError: jid += 1
    record = {'id': jid, 'job_kind': 'native_dependency', 'name': 'isg261110_ik_probe',
              'script': str(Path(value['helper_snapshot']) / 'scripts/campaign_20261110/job_ik_native_probe.sh'),
              'job_dir': str(job), 'time_s': 2400, 'cpus': 8, 'mem': '48G', 'acct': 'vision', 'part': 'svl',
              'log': str(root / 'slurm' / ('isg261110_ik_probe_' + str(jid) + '.log')),
              'cwd': str(root), 'submitted': time.time(), 'submitted_by': 'campaign_exact_ik_probe',
              'gpu': gpu, 'native_inputs_sha256': sha(job / 'native_inputs.json'),
              'exact_ik_exchange_required': True, 'diagnostic_only': True, 'benchmark_credit': 0}
    target = root / 'vast/queue' / (str(jid) + '.json')
    write(target, record); write(receipt, {'id': jid, 'record_sha256': sha(target), 'activated': False})
    return str(jid)


def run(job):
    value = verify(job)
    job = Path(job).resolve(); output = Path(value['probe_output'])
    # Worker lifecycle owns these transport paths; all physics settings retain
    # the exact recorded source environment and are checked again by runtime.
    transport_keys = ('ISPATIALGYM_EXACT_IK_LOCAL_CACHE', 'ISPATIALGYM_EXACT_IK_PRODUCER',
                      'ISPATIALGYM_EXACT_IK_METRICS')
    transport = {k: os.environ[k] for k in transport_keys if k in os.environ}
    os.environ.update(value['environment']); os.environ.update(transport)
    helper = Path(value['helper_snapshot']) / 'scripts/campaign_20261110/ik_native_probe.py'
    cmd = [sys.executable, str(helper), '--spec', value['probe_spec'], '--output', str(output)]
    with (job / 'probe.log').open('w') as log:
        try: rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=1800).returncode
        except subprocess.TimeoutExpired: rc = 124
    if output.exists(): shutil.copytree(output, job / 'result')
    report = json.loads((output / 'probe_report.json').read_text()) if (output / 'probe_report.json').exists() else {}
    actual_tros = list(output.rglob('*-tro_state.json')) if output.exists() else []
    good = rc == 0 and report.get('sampler_exit_code') in (None, 0) and not actual_tros and report.get('published_tro_files') == 0 and report.get('status') in (
        'bounded_prefix_complete', 'one_draw_finished')
    write(job / 'probe_completion.json', {'campaign_id': value['campaign_id'], 'returncode': rc,
          'diagnostic_only': True, 'benchmark_credit': 0, 'source_output': str(output),
          'copied_output': str(job / 'result'), 'finished_at': time.time(), 'complete': good,
          'actual_published_tro_files': len(actual_tros),
          'payload_files': {str(p.relative_to(job / 'result')): sha(p)
                            for p in (job / 'result').rglob('*') if p.is_file()}})
    return 0 if good else 2


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--verify', type=Path); p.add_argument('--run', type=Path)
    a = p.parse_args()
    if a.verify: verify(a.verify)
    elif a.run: sys.exit(run(a.run))
    else: p.error('choose --verify or --run')
