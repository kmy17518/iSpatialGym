#!/usr/bin/env python3
"""Stream each physics-validated positive scale library into measured-height maps."""
import argparse
import datetime
import json
import os
import pathlib
import subprocess
import time


def read(path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path, required=True)
    parser.add_argument('--snapshot', type=pathlib.Path, required=True)
    parser.add_argument('--first', type=int, required=True)
    parser.add_argument('--last', type=int, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    root = args.root
    output = root / 'scale_prebuild'
    state_path = output / 'controller.json'
    state = read(state_path, {'entries': {}})
    manifest = root / 'manifests/train_scale_glib_jobs.json'
    jobs = read(manifest, [])
    assert len(jobs) == 54
    anchors = root / 'manifests/train_scale_height_anchors.json'
    heights = root / 'manifests/train_scale_heights_by_id.json'
    height_keys = {str(x['key']) for x in read(anchors, {})['anchors']}
    by_id = read(heights, {})
    while True:
        queue = subprocess.run(['squeue', '-h', '-r', '-u', os.environ['USER'], '-o', '%i|%j|%T'], capture_output=True, text=True, check=True)
        active = {line.split('|')[0] for line in queue.stdout.splitlines()}
        for index in range(args.first, args.last + 1):
            job = jobs[index]
            entry = state['entries'].setdefault(job['id'], {'index': index, 'status': 'waiting_for_library'})
            log = output / f'logs/glib_{index}.log'
            if not log.exists() or 'GB_VALIDATE_DONE' not in log.read_text(errors='replace'):
                continue
            library = read(output / 'glib/models' / job['id'] / 'library.json', {})
            entry['valid_grasps'] = len(library.get('grasps', []))
            if not entry['valid_grasps']:
                entry['status'] = 'no_valid_grasps'
                continue
            assert library.get('scale') == job['scale'], job['id']
            selected = by_id.get(job['id'], [])
            if not selected or not all(str(key) in height_keys for key in selected):
                entry['status'] = 'missing_measured_height'
                continue
            if entry.get('status') == 'submitting' and not entry.get('map_job_id'):
                entry['status'] = 'submission_interrupted_needs_audit'
                continue
            if entry.get('status') == 'submission_interrupted_needs_audit':
                continue
            if entry.get('map_job_id'):
                if entry['map_job_id'] in active or entry['map_job_id'] + '_' + str(index) in active:
                    entry['status'] = 'maps_active'
                else:
                    records = []
                    for arm in ('right', 'left'):
                        path = output / f'logs/maps_{index}_{arm}.log'
                        if path.exists():
                            records.extend(json.loads(line.split(' ', 1)[1]) for line in path.read_text(errors='replace').splitlines() if line.startswith('GB_MAP {'))
                    entry['map_records'] = len(records)
                    entry['map_ok_records'] = sum(row.get('ok') is True for row in records)
                    entry['status'] = 'maps_finished_unreleased' if records else 'maps_waiting_or_missing'
                continue
            env = dict(os.environ)
            env.update(GBH=str(root), ISG_GB_ROOT=str(args.snapshot), GB_OUT=str(output), GB_MANIFEST=str(manifest), GB_NSHARD='54', GB_TAG='train_scales', GB_STAGES='maps', GB_HEIGHTS_BY_ID=str(heights), GB_ANCHORS=str(anchors), GB_ARMS='right left', GB_INCLUDE_DERIVED='0', KIT_TPL_NFS_ROOT=str(root / 'kit_templates'), KIT_TEX_STORE=str(root / 'kit_textures'), EXP_PATH=str(root / 'isaacsim_overlay/apps'), GIT_CONFIG_COUNT='1', GIT_CONFIG_KEY_0='safe.directory', GIT_CONFIG_VALUE_0='*')
            command = ['sbatch', '--parsable', '-J', 'isg261110_maps', '-A', 'vision', '-p', 'svl', '--exclude=svl9', '--gres=gpu:titanrtx:1', '-c', '8', '--mem=40G', '--time=04:00:00', '--array=' + str(index), '-o', str(root / 'slurm/%x_%A_%a.log'), '--export=ALL', str(args.snapshot / 'scripts/grasp_bands/job_glib.sh')]
            entry['status'] = 'submitting'
            save(state_path, state)
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=45)
            if result.returncode:
                entry.update(status='submission_failed', error=result.stderr[:300])
            else:
                entry.update(status='maps_submitted', map_job_id=result.stdout.strip().split(';')[0], submitted_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
                print('MAP_SUBMITTED', job['id'], entry['map_job_id'], flush=True)
            save(state_path, state)
        state['updated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        state['summary'] = {key: sum(entry['status'] == key for entry in state['entries'].values()) for key in sorted({entry['status'] for entry in state['entries'].values()})}
        save(state_path, state)
        if args.once:
            break
        time.sleep(30)


if __name__ == '__main__':
    main()
