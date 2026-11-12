"""Archive a stopped worker's existing outputs without starting collection.

This module runs before GPU initialization and before the normal worker. It
never imports simulation code, edits a source job, emits acceptance, or destroys
an instance. The coordinator separately verifies the uploaded archive.
"""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

CAMPAIGN_ROOT = Path('/vision/u/minyeong/projects/isg_home/campaign_20261110')
SCHEMA = 'isg-private-worker-boot-policy-v1'


def policy(path, instance):
    value = json.loads(Path(path).read_text())
    if value.get('schema') != SCHEMA or value.get('campaign_id') != 'campaign_20261110':
        raise ValueError('worker boot policy identity differs')
    return value.get('recover_instances', {}).get(str(instance))


def verified_job(row, root=CAMPAIGN_ROOT):
    job = Path(row['job_dir']).resolve()
    if not job.is_relative_to(root / 'out') or not str(row['job_id']).isdigit():
        raise ValueError('recovery job outside exact campaign output')
    if row.get('kind') == 'tabletop_maps':
        manifest = job/'native_inputs.json'
        if not job.is_relative_to(root/'out/vast_map_prebuild') or digest(manifest) != row['native_inputs_sha256']:
            raise ValueError('map recovery manifest differs from stopped assignment')
        value = json.loads(manifest.read_text())
        if (value.get('schema') != 'isg-vast-map-prebuild-v1' or value.get('campaign_root') != str(root)
                or value.get('job_dir') != str(job) or value.get('kind') != 'tabletop_maps'
                or value.get('new_native_instances') != 0 or value.get('production_released') is not False):
            raise ValueError('map recovery purpose differs')
        for name, expected in {**value['files'], **value['library_files'], **value.get('seed_map_files', {})}.items():
            path = Path(name).resolve()
            if not path.is_relative_to(root) or digest(path) != expected:
                raise ValueError('map recovery input changed')
        return job
    if row.get('kind') not in (None, 'training'):
        raise ValueError('unknown recovery job kind')
    env = {}
    for line in (job / 'job.env').read_text().splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            words = shlex.split(value) if value else ['']
            if len(words) == 1: env[key] = words[0]
    config = Path(env['ISPATIALGYM_CAMPAIGN_CONFIG']).resolve()
    if not config.is_relative_to(root): raise ValueError('recovery config outside campaign')
    raw = config.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row['config_sha256']:
        raise ValueError('recovery config differs from stopped assignment')
    if json.loads(raw)['template']['split'] != 'train':
        raise ValueError('only proven stopped standalone training may be recovered')
    return job


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''): result.update(block)
    return result.hexdigest()


def s5(*args):
    result = subprocess.run(['s5cmd', '--endpoint-url', os.environ['S3_ENDPOINT_URL'], *args],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=900)
    if result.returncode: raise RuntimeError('private recovery transfer failed')


def recover(value, instance, *, root=CAMPAIGN_ROOT, work_root=Path('/work/training_gate_recovery')):
    purposes = {'preserve_stopped_training_artifacts': (None, 'training'),
                'preserve_stopped_map_artifacts': ('tabletop_maps',)}
    if value.get('purpose') not in purposes or not value.get('jobs'):
        raise ValueError('invalid bounded recovery purpose')
    if any(row.get('kind') not in purposes[value['purpose']] for row in value['jobs']):
        raise ValueError('recovery job kind differs from authorized purpose')
    prefix = os.environ['ISG_S3'].rstrip('/')
    if not prefix.endswith('/jobs/og-isg261110'): raise ValueError('recovery storage scope differs')
    work = Path(work_root) / str(instance)
    work.mkdir(parents=True, exist_ok=True)
    report = {'campaign_id':'campaign_20261110', 'instance':str(instance),
              'purpose':value['purpose'], 'started_unix':time.time(), 'collection_started':False,
              'source_jobs_modified':False, 'jobs':[]}
    for row in value['jobs']:
        job = verified_job(row, root=root)
        archive = work / (row['job_id'] + '.tar.zst')
        command = ['tar', '-C', '/', '--format=posix', '--exclude=*.usd', '--exclude=*.usda',
                   '--exclude=*.usdc', '--exclude=*.encrypted*', '-cf', '-', str(job).lstrip('/')]
        tar = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        zip_result = subprocess.run(['zstd','-q','-T2','-3','-f','-o',str(archive)], stdin=tar.stdout,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
        tar.stdout.close(); tar.stdout = None; tar_error = tar.communicate(timeout=60)[1]
        if tar.returncode or zip_result.returncode: raise RuntimeError('recovery archive failed')
        key = prefix + '/recovery/training_gate_20261111/' + str(instance) + '/' + archive.name
        s5('cp', str(archive), key)
        native_paths = (job/'result/maps').rglob('map__*') if row.get('kind') == 'tabletop_maps' else (job/'inst').rglob('*-tro_state.json')
        native = [{'path':str(p), 'sha256':digest(p), 'bytes':p.stat().st_size}
                  for p in native_paths if p.is_file()]
        report['jobs'].append({**row, 'archive_key':key, 'archive_sha256':digest(archive),
                               'archive_bytes':archive.stat().st_size, 'native_files':native})
    report['finished_unix'] = time.time()
    path = work/'report.json'; path.write_text(json.dumps(report,indent=2)+'\n')
    s5('cp',str(path),prefix+'/recovery/training_gate_20261111/'+str(instance)+'/report.json')
    print(json.dumps({'instance':str(instance),'recovery_only':True,'jobs':len(report['jobs']),
                      'report_sha256':digest(path)}),flush=True)


if __name__ == '__main__':
    mode, path, instance = sys.argv[1:]
    value = policy(path, instance)
    if mode == 'mode': print('recover' if value is not None else 'normal')
    elif mode == 'recover':
        if not value: raise ValueError('no exact recovery authorization')
        recover(value, instance)
    else: raise ValueError('unknown recovery mode')
