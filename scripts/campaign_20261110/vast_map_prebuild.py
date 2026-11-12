"""Exact native-library/height reach-map preparation on private campaign workers."""
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

# The worker executes this file by path while native modules retain their frozen PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.campaign_20261110.vast_prebuild import require, sha, write

SCHEMA = 'isg-vast-map-prebuild-v1'
DENIED_SUFFIXES = {'.pt', '.pth', '.ckpt', '.safetensors'}
SECONDARY_PROOF_ROOT = Path('/vision/u/shiyuc/minyeong/campaign_20261110/training_preparation')


def measured_records(unit):
    dependencies = Path(unit['source_dependencies']['path'])
    require(sha(dependencies) == unit['source_dependencies']['sha256'], 'map source dependencies changed')
    source = json.loads(dependencies.read_text())
    records = [r for r in source['requests'] if r['entry_id'] == unit['job']['id']
               and r['actual_height_key'] == unit['height'] and r['library'] == unit['source_library']]
    require(records, 'exact measured source missing')
    return records


def secondary_proof_path(path):
    path = Path(path)
    return (path.is_absolute() and '..' not in path.parts and path.suffix == '.json'
            and path.is_relative_to(SECONDARY_PROOF_ROOT))


def copy_measured_proofs(root, job, records):
    files, relocations = [], {}
    for record in records:
        for key in ('probe', 'stable_scene'):
            ref = record[key]; src = Path(ref['path'])
            require(sha(src) == ref['sha256'], 'map proof changed: '+key)
            if src.resolve().is_relative_to(root):
                files.append(src)
                continue
            require(secondary_proof_path(src) and not src.is_symlink() and src.is_file()
                    and src.resolve().is_relative_to(SECONDARY_PROOF_ROOT.resolve()),
                    'measured height proof outside approved campaign preparation')
            dst = job/'proof_inputs'/(ref['sha256']+'.json')
            dst.parent.mkdir(exist_ok=True)
            if not dst.exists(): shutil.copyfile(src, dst)
            require(sha(dst) == ref['sha256'] and sha(src) == ref['sha256'], 'measured proof copy changed')
            relocations[str(src)] = {'copy': str(dst), 'sha256': ref['sha256']}
            files.append(dst)
    return files, relocations


def library_environment(root, v2_root=None):
    return {}


def prepare(root, plan, index, native, helper, token):
    root, plan, native, helper = map(lambda p: Path(p).resolve(), (root, plan, native, helper))
    require(re.fullmatch(r'[A-Za-z0-9_-]+', token), 'unsafe map token')
    require(plan.is_relative_to(root / 'training_preparation'), 'map plan outside campaign preparation')
    doc = json.loads(plan.read_text())
    require(doc['runtime'] == str(native) and native.is_relative_to(root/'code') and helper.is_relative_to(root/'code'), 'frozen map runtime differs')
    require(type(index) is int and 0 <= index < len(doc['units']), 'invalid map unit')
    unit = doc['units'][index]
    require(unit['id'] == index and unit['arms'] and set(unit['arms']) <= {'right', 'left'}, 'invalid map arms')
    row = unit['job']
    require(row.get('ready') is True and row['kind'] == 'tabletop' and row.get('constraint') == 'tabletop', 'unsupported library kind')
    require(not any(c.startswith('configuration_matching-block_design') for c in row.get('campaign_classes', [])), 'selection remains under review')
    dependencies = Path(unit['source_dependencies']['path'])
    require(dependencies.is_relative_to(root) and sha(dependencies) == unit['source_dependencies']['sha256'], 'map source dependencies changed')
    matches = measured_records(unit)
    for record in matches:
        for key in ('library', 'probe', 'stable_scene'):
            require(sha(record[key]['path']) == record[key]['sha256'], 'map proof changed: '+key)
    library = Path(unit['source_library']['path']).resolve()
    require(library.name == 'library.json' and library.parent.parent.name == 'models', 'native library closure required')
    anchors = Path(unit['anchors'])
    require(anchors.is_relative_to(root) and str(unit['height']) in {a['key'] for a in json.loads(anchors.read_text())['anchors']}, 'exact height anchor missing')
    job = root/'out/vast_map_prebuild'/token
    require(not job.exists(), 'map token already has evidence')
    job.mkdir(parents=True)
    copied = job/'inputs/glib/models'/library.parent.name
    copied.mkdir(parents=True)
    relocation = {}
    for src in library.parent.rglob('*'):
        if not src.is_file(): continue
        require(src.resolve().is_relative_to(library.parent) and src.suffix.lower() not in DENIED_SUFFIXES, 'only native geometry/library closure is allowed')
        dst = copied/src.relative_to(library.parent)
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dst)
        require(sha(src) == sha(dst), 'native library copy changed')
        relocation[str(src)] = {'copy': str(dst), 'sha256': sha(dst)}
    selected = job/'selected_jobs.json'; write(selected, [row])
    private_anchors = job/'anchors.json'; shutil.copyfile(anchors, private_anchors)
    proof_files, proof_relocations = copy_measured_proofs(root, job, matches)
    seed_files = {}
    for record in unit.get('existing_maps', []):
        original = Path(record['path'])
        require(sha(original) == record['sha256'] and sha(original.with_suffix('.npz')) == record['npz_sha256'], 'existing map changed')
        for src in (original, original.with_suffix('.npz')):
            dst = job/'seed_maps'/row['id']/src.name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dst); seed_files[str(dst)] = sha(src)
    files = proof_files + [plan, dependencies, selected, private_anchors, job/'job.env', native/'.snapshot_commit',
             native/'scripts/grasp_bands/job_glib.sh', helper/'scripts/campaign_20261110/vast_map_prebuild.py',
             helper/'scripts/campaign_20261110/vast_prebuild.py', helper/'scripts/campaign_20261110/job_map_prebuild.sh']
    env = {'ISGE_CODE': str(native), 'ISGF_ROOT': str(root), 'ISGF_NATIVE_HELPER': str(helper)}
    (job/'job.env').write_text(''.join(k+'='+shlex.quote(v)+'\n' for k,v in env.items()))
    value = {'schema': SCHEMA, 'campaign_id': root.name, 'campaign_root': str(root), 'job_dir': str(job),
             'snapshot': str(native), 'helper_snapshot': str(helper), 'source_plan': str(plan), 'unit_index': index,
             'unit': unit, 'selected_manifest': str(selected), 'private_glib': str(copied.parent.parent),
             'seed_map_files': seed_files, 'anchors': str(private_anchors), 'files': {str(p): sha(p) for p in files},
             'library_files': {v['copy']: v['sha256'] for v in relocation.values()}, 'library_relocations': relocation,
             'proof_relocations': proof_relocations,
             'original_entry': {**row, 'class_id': 'dependency-tabletop_maps', 'task': 'reach_map_preparation'},
             'kind': 'tabletop_maps', 'scenes': [], 'time_limit_s': 9000,
             'new_native_instances': 0, 'benchmark_credit': 0, 'production_released': False}
    write(job/'native_inputs.json', value)
    (job/'sample.log').symlink_to('result/native_'+unit['arms'][0]+'.log')
    return job


def verify(job, root=None):
    job = Path(job).resolve(); value = json.loads((job/'native_inputs.json').read_text())
    root = Path(root or value['campaign_root']).resolve()
    require(value['schema'] == SCHEMA and value['campaign_root'] == str(root), 'map campaign differs')
    require(job.is_relative_to(root/'out/vast_map_prebuild') and value['job_dir'] == str(job), 'map output ownership differs')
    require(value['kind'] == 'tabletop_maps' and value['new_native_instances'] == 0 and value['production_released'] is False, 'map purpose differs')
    for name, expected in {**value['files'], **value['library_files'], **value.get('seed_map_files', {})}.items():
        require(Path(name).resolve().is_relative_to(root) and sha(name) == expected, 'map input changed: '+name)
    doc = json.loads(Path(value['source_plan']).read_text()); unit = doc['units'][value['unit_index']]
    require(value['unit'] == unit and doc['runtime'] == value['snapshot'], 'map unit changed')
    dependencies = Path(unit['source_dependencies']['path'])
    require(dependencies.resolve().is_relative_to(root) and value['files'].get(str(dependencies))
            == unit['source_dependencies']['sha256'], 'map measured source is not bound')
    expected_relocations = {}
    for record in measured_records(unit):
        for key in ('probe', 'stable_scene'):
            ref = record[key]; original = Path(ref['path'])
            if original.is_relative_to(root):
                require(value['files'].get(str(original)) == ref['sha256'], 'map measured proof is not bound')
            else:
                require(secondary_proof_path(original), 'map proof relocation source escaped')
                copied = job/'proof_inputs'/(ref['sha256']+'.json')
                expected_relocations[str(original)] = {'copy': str(copied), 'sha256': ref['sha256']}
                require(not copied.is_symlink() and copied.resolve().is_relative_to(job/'proof_inputs')
                        and value['files'].get(str(copied)) == ref['sha256'], 'map proof relocation is not bound')
    require(value.get('proof_relocations', {}) == expected_relocations, 'map proof relocation differs from measured source')
    require(json.loads(Path(value['selected_manifest']).read_text()) == [unit['job']], 'map native row changed')
    require(set(unit['arms']) <= {'right','left'} and unit['arms'], 'map arms changed')
    require(all(Path(p).is_relative_to(job/'inputs/glib') and Path(p).suffix.lower() not in DENIED_SUFFIXES for p in value['library_files']), 'map library closure escaped')
    return value


def enqueue(job, root, gpu='RTX 4090'):
    root, job = Path(root).resolve(), Path(job).resolve(); value = verify(job, root)
    receipt = job/'vast_submission.json'
    if receipt.exists():
        old = json.loads(receipt.read_text()); require(sha(root/'vast/queue'/(str(old['id'])+'.json')) == old['record_sha256'], 'map queue changed'); return str(old['id'])
    ids = root/'vast/queue/ids'; ids.mkdir(parents=True, exist_ok=True)
    jid = max([900000000]+[int(p.name) for p in ids.iterdir() if p.name.isdigit()])+1
    while True:
        try: (ids/str(jid)).mkdir(); break
        except FileExistsError: jid += 1
    record = {'id': jid, 'job_kind': 'native_dependency', 'name': 'isg261110_map_prebuild',
              'script': str(Path(value['helper_snapshot'])/'scripts/campaign_20261110/job_map_prebuild.sh'),
              'job_dir': str(job), 'time_s': 9600, 'cpus': 9, 'mem': '48G', 'acct': 'vision', 'part': 'svl',
              'log': str(root/'slurm'/('map_prebuild_'+str(jid)+'.log')), 'cwd': str(root),
              'submitted': time.time(), 'submitted_by': 'typed_map_prebuild', 'gpu': gpu,
              'native_inputs_sha256': sha(job/'native_inputs.json')}
    target = root/'vast/queue'/(str(jid)+'.json'); write(target, record)
    write(receipt, {'id': jid, 'record_sha256': sha(target), 'activated': False}); return str(jid)


def run(job):
    job = Path(job).resolve(); value = verify(job); unit = value['unit']; output = job/'result'; output.mkdir(exist_ok=False)
    if (job/'seed_maps').exists(): shutil.copytree(job/'seed_maps', output/'maps')
    start = time.time(); rows = []
    for arm in unit['arms']:
        live = job/'sample.log'
        if live.is_symlink(): live.unlink()
        live.symlink_to('result/native_'+arm+'.log')
        argv = [sys.executable, '-u', '-m', 'ispatialgym.sampling.reach_maps', 'compute', '--kind', 'tabletop',
                '--jobs', value['selected_manifest'], '--glib', value['private_glib'], '--anchors', value['anchors'],
                '--heights', unit['height'], '--arm', arm, '--out', str(output/'maps'), '--skip-existing']
        logpath = output/('native_'+arm+'.log'); write(output/('command_'+arm+'.json'), {'argv': argv, 'started_at': time.time(), 'snapshot': value['snapshot']})
        with logpath.open('w') as log:
            try: rc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=max(1,value['time_limit_s']-(time.time()-start))).returncode
            except subprocess.TimeoutExpired: rc = 124
        text = logpath.read_text(errors='replace'); done = 'GB_MAP_DONE' in text
        rows.append({'arm': arm, 'returncode': rc, 'native_map_done': done, 'log': str(logpath)})
        write(output/'progress.json', {'arms': rows, 'observed_at': time.time()})
    maps = sorted((output/'maps').rglob('map__*.json'))
    report = {'arms': rows, 'map_files': {str(p): sha(p) for p in maps}, 'positive_native_reader_required': True,
              'started_at': start, 'finished_at': time.time(), 'new_native_instances': 0, 'benchmark_credit': 0}
    write(output/'map_prebuild_result.json', report)
    return 0 if maps and all(r['returncode']==0 and r['native_map_done'] for r in rows) else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--verify'); parser.add_argument('--run'); args = parser.parse_args()
    if args.verify: verify(args.verify); print('exact map dependency inputs verified')
    elif args.run: sys.exit(run(args.run))
    else: parser.error('choose --verify or --run')
