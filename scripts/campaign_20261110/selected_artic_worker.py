"""Run and reconcile one exact selected motion component; never collect instances."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

CAMPAIGN_ROOTS = (
    Path('/vision/u/minyeong/projects/isg_home/campaign_20261110'),
    Path('/vision/u/shiyuc/minyeong/campaign_20261110'),
)


def read(path):
    return json.loads(Path(path).read_text())


def ref(path):
    return {'path': str(path), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def write(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def verify(path, index):
    wave = read(path)
    assert wave['schema'] == 'isg-selected-artic-motion-wave-v1'
    assert wave['campaign_id'] == 'campaign_20261110' and wave['production_authorization'] is False
    unit = wave['units'][index]
    assert unit['index'] == index and unit['benchmark_credit'] == unit['new_native_instances'] == 0
    for r in [wave['source_handoff'], wave['selected_manifest'], unit['request'], unit['source_customs'], unit['customs'],
              unit['identity']['geometry'], unit['identity']['handle_annotation']]:
        assert ref(r['path']) == r, r['path']
    runtime = Path(unit['native_runtime'])
    for name, sha in unit['identity']['generator_files'].items():
        assert ref(runtime / name)['sha256'] == sha, name
    assert runtime == Path(wave['native_runtime'])
    return wave, unit


def resolve_output(path, wave, unit, output_root=None):
    """Relocate output only, preserving the immutable wave and native unit."""
    original = Path(unit['output'])
    if output_root is None:
        return original, None
    source = Path(path).resolve()
    root = Path(output_root)
    assert root.is_absolute() and '..' not in root.parts, 'output root must be an absolute campaign path'
    assert re.fullmatch(r'artic_[0-9a-f]{24}', unit['id']), 'invalid physical unit identity'
    # The caller may relocate only into this same named preparation wave.
    assert source.parent.name.startswith('selected_artic_motion_'), 'foreign request scope'
    assert any(source.is_relative_to(c / 'training_preparation') for c in CAMPAIGN_ROOTS), 'foreign request root'
    allowed = [c / 'training_preparation' / source.parent.name for c in CAMPAIGN_ROOTS]
    resolved = root.resolve()
    assert any(root.is_relative_to(p) and resolved.is_relative_to(p.resolve()) for p in allowed), 'output root outside isolated campaign wave'
    assert original.resolve().is_relative_to(source.parent), 'original output outside request wave'
    out = resolved / unit['id']
    assert out.resolve().is_relative_to(resolved), 'unit output escaped relocated root'
    mapping = {'schema': 'isg-selected-artic-output-relocation-v1', 'campaign_id': wave['campaign_id'],
               'request': ref(path), 'unit_request': unit['request'], 'unit_id': unit['id'], 'unit_index': unit['index'],
               'unit_payload_sha256': hashlib.sha256(json.dumps(unit, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
               'original_output': str(original), 'output_root': str(resolved), 'actual_output': str(out),
               'native_runtime': unit['native_runtime'], 'native_inputs_unchanged': True,
               'worker': ref(__file__), 'benchmark_credit': 0, 'production_authorization': False}
    return out, mapping


def audit_library(path, unit):
    from ispatialgym.grasping import schema as S
    from ispatialgym.grasping import validate_artic as VA
    lib = read(path)
    identity = unit['identity']
    assert lib['schema'] == VA.LIBRARY_V2 and lib['alias'] == unit['alias']
    assert (lib['category'], lib['model'], lib['joint']) == (identity['category'], identity['model'], identity['joint'])
    assert S.canonical_scale(lib['scale']) == S.canonical_scale(identity['scale'])
    assert lib['arm_validated'] == identity['arm'] and lib['n_segments'] == len(lib['segments'])
    assert lib['motion'] == ('push_close' if identity['generation']['direction'] == 'close' else 'pull_open')
    assert lib['config'] == json.loads(json.dumps(__import__('dataclasses').asdict(VA.MotionConfig())))
    for segment in lib['segments']:
        assert segment['arm_validated'] == identity['arm'] and segment['joint'] == identity['joint']
        assert segment['trial_stop'] not in VA.HARD_STOPS and segment['validated_travel_rad'] >= VA.MotionConfig.min_travel_rad
        assert segment['waypoints'] and segment['handle_sig'] == lib['handle_sig']
        rest = lib['rests'][segment['rest_id']]
        assert rest['stable'] is True
        S.Rest.from_record(rest)
        validation = (path.parent / segment['source']['validation']).resolve()
        assert validation.is_relative_to(path.parent.resolve()) and validation.is_file()
        trace = read(validation)
        assert trace['validation_world']['arm'] == identity['arm']
        assert trace['config']['artic']['fixed_base'] is False
        assert trace['motion']['handle_sig'] == lib['handle_sig']
        assert S.canonical_scale(trace['motion']['scale']) == S.canonical_scale(identity['scale'])
    moving = sum(s['validated_travel_rad'] > 0 and len(s['waypoints']) >= 2 for s in lib['segments'])
    return {'artifact': ref(path), 'segments': lib['n_segments'], 'positive_travel_segments': moving,
            'q_start': lib['q_start'], 'motion': lib['motion'], 'arm': lib['arm_validated'],
            'handle_sig': lib['handle_sig'], 'positive': moving > 0}


def run(path, index, output_root=None):
    wave, unit = verify(path, index)
    out, relocation = resolve_output(path, wave, unit, output_root)
    if relocation is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)
    if relocation is not None:
        write(out / 'output_relocation.json', relocation)
    start = time.time()
    result = {'schema': 'isg-selected-artic-motion-result-v1', 'campaign_id': 'campaign_20261110', 'unit_id': unit['id'],
              'request': ref(path), 'unit_request': unit['request'], 'owner': '/root', 'job_id': os.environ.get('SLURM_JOB_ID'),
              'array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID'), 'started_at': start, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
              'benchmark_credit': 0, 'production_authorization': False, 'status': 'running', 'artifacts': []}
    if relocation is not None:
        result['output_relocation'] = ref(out / 'output_relocation.json')
    result['actual_output'] = str(out)
    result['original_output'] = unit['output']
    write(out / 'result.json', result)
    runtime = Path(unit['native_runtime'])
    sys.path.insert(0, str(runtime))
    log = out / 'native.log'
    cmd = [sys.executable, '-u', '-m', 'ispatialgym.grasping.validate_artic', 'motions', '--aliases', unit['alias'],
           '--root', str(out / 'alib'), '--arm', unit['identity']['arm'], '--q-starts', 'grid', '--customs', unit['customs']['path']]
    result['command'] = cmd
    try:
        with log.open('w') as f:
            try:
                process = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=2400, cwd=runtime)
                result['returncode'] = process.returncode
            except subprocess.TimeoutExpired:
                result['returncode'] = 'timeout_2400s'
        result['log'] = ref(log)
        done = 'GB_ARTIC_DONE' in log.read_text(errors='replace')
        result['native_done_marker'] = done
        from ispatialgym.grasping import object_geometry as OG, schema as S
        model = Path(S.model_dir(unit['identity']['category'], unit['identity']['model'], unit['identity']['scale'], str(out / 'alib'), 'articulation'))
        libraries = sorted(model.glob('artic_library__*.json'))
        if done and libraries:
            geom = OG.load(str(model))
            assert geom is not None and len(geom.union.vertices) > 0
            result['artifacts'] = [audit_library(p, unit) for p in libraries]
        positive = [r for r in result['artifacts'] if r['positive']]
        result['status'] = 'verified_motion_components' if positive else 'failed_no_verified_positive_motion'
        result['positive_libraries'] = len(positive)
        result['consumer_configuration_ids'] = unit['consumer_configuration_ids']
        result['logical_request_ids'] = unit['logical_request_ids']
        result['next_action'] = ('Stream each positive exact motion/start into measured-height prefix maps; complete configuration remains unverified.'
                                 if positive else 'Read native rest/trial failure evidence and make a targeted changed proposal; no unchanged retry.')
    except Exception as exc:
        result.update(status='failed_verification', error=repr(exc), next_action='Root diagnoses exact artifact/trace mismatch; do not promote.')
    finally:
        result.update(finished_at=time.time(), elapsed_seconds=time.time()-start)
        write(out / 'result.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--unit', type=int, required=True)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--output-root', type=Path, help='Relocate outputs within this named campaign preparation wave only')
    args = parser.parse_args()
    if args.verify_only:
        wave, unit = verify(args.request, args.unit)
        out, relocation = resolve_output(args.request, wave, unit, args.output_root)
        print(json.dumps({'status': 'inputs_verified', 'unit_id': unit['id'], 'request': ref(args.request),
                          'actual_output': str(out), 'output_relocation': relocation}))
    else:
        run(args.request, args.unit, args.output_root)
