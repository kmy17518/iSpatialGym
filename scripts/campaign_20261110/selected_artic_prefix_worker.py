"""Build exact measured-height articulation prefixes and verify returned maps."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def ref(path):
    return {'path': str(path), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def verify_ref(value):
    assert ref(value['path'])['sha256'] == value['sha256'], value['path']
    if value.get('relocation_proof'):
        verify_ref(value['relocation_proof'])


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def verify(request, index):
    wave = read(request)
    assert wave['schema'] == 'isg-selected-artic-prefix-wave-v1' and wave['campaign_id'] == 'campaign_20261110'
    assert wave['benchmark_credit'] == 0 and wave['production_authorization'] is False
    unit = wave['units'][index]
    assert unit['index'] == index and unit['native_runtime'] == wave['native_runtime']
    for proof in [wave['selected_manifest'], wave['motion_wave'], unit['request'], unit['anchors'], unit['motion_request'], unit['motion_result']]:
        verify_ref(proof)
    for proof in unit['identity']['libraries']:
        verify_ref(proof)
    for consumer in unit['consumers']:
        verify_ref(consumer['support_artifact'])
        assert abs(consumer['measured_height'] - float(unit['identity']['height_key'])) <= .00005
    for name, sha in unit['identity']['generator_files'].items():
        verify_ref({'path': str(Path(unit['native_runtime']) / name), 'sha256': sha})
    return wave, unit


def audit_map(path, unit):
    import numpy as np
    from ispatialgym.grasping import schema as S, contact_policy as CP
    from ispatialgym.sampling import reach_maps as RM, height_anchors as HA, reach_bands as RB

    record = RM.PrefixMap(str(path))
    meta = record.meta
    identity = unit['identity']
    source = next((r for r in identity['libraries'] if r['path'] == meta.get('source', {}).get('library')), None)
    assert source is not None, 'map names a different source motion'
    verify_ref(source)
    lib = read(source['path'])
    assert meta['key'] == RM.prefix_map_key(meta)
    assert meta['source']['library_digest'] == S.digest({k: lib.get(k) for k in ('alias', 'segments', 'rests')}, 16)
    assert meta['arm_validated'] == [identity['arm']] and meta['arm'] == identity['arm']
    assert meta['segment_ids'] == [s['id'] for s in lib['segments']]
    assert abs(meta['q_start'] - lib['q_start']) <= .00005
    assert meta['motion'] == lib['motion'] and meta['handle_sig'] == lib['handle_sig']
    assert float(meta['height']['value']) == float(identity['height_key'])
    assert meta['policy_key'] == CP.DEFAULT.key(identity['arm']) and meta['robot'] == RM.robot_digest()
    cfg = RM.MapConfig(arm=identity['arm'], lattice_m=.08, seeds=8, batch=2048, twins=False)
    assert meta['config'] == json.loads(json.dumps(cfg.key()))
    assert meta['edge_distances'] == list(cfg.edge_distances)
    assert meta['min_open_rad'] == 0 and meta['step_max_rad'] == .5 and meta['yaw_offsets'] == [0.]
    assert np.array_equal(record.cells, RM.lattice(cfg).astype(np.float32))
    assert np.isfinite(record.yaws).all() and len(record.yaws) == 1
    face = float(lib['face_yaw_offset'])
    assert np.allclose(record.yaws, [math.atan2(math.sin(face), math.cos(face))], rtol=0, atol=1e-6)
    expected_trunks = {(float(a), float(b)) for a, b in RB.trunk_grid(float(identity['height_key']))}
    actual_trunks = [(float(t['translate']), float(t['tilt'])) for t in meta['trunks']]
    assert len(actual_trunks) == len(expected_trunks) and set(actual_trunks) == expected_trunks
    positive = 0
    expected = {'cells', 'yaws'}
    for trunk in meta['trunks']:
        for edge in meta['edge_distances']:
            for i, segment in enumerate(lib['segments']):
                name = f's{edge:.2f}_{trunk["key"]}__seg{i}'
                expected.add(name)
                values = record.prefix(edge, trunk['translate'], trunk['tilt'], i)
                assert values.shape == (len(record.cells), len(record.yaws)) and np.isfinite(values).all()
                assert values.min() >= -1.
                # Bound against unrounded source waypoints; display metadata is
                # rounded to four places and must not redefine motion travel.
                q = [float(w['q']) for w in segment['waypoints']]
                travel = max(abs(v-q[0]) for v in q)
                assert values.max() <= travel + 1e-5
                positive += int((values > 1e-4).sum())
    assert set(record.arrays) == expected
    rest = lib['rests'][lib['segments'][0]['rest_id']]
    spec = {k: lib[k] for k in ('category', 'model', 'scale', 'joint', 'motion', 'q_start', 'handle_sig', 'face_yaw_offset')}
    spec.update(arm=identity['arm'], height=float(identity['height_key']), rest_quat=rest['intended_quat_xyzw'],
                needed_travel=0., min_open_rad=0., step_max_rad=.5)
    anchors = HA.AnchorSet(read(unit['anchors']['path'])['anchors'], max_approx_m=0)
    found = RM.find_prefix(str(path.parent), spec, anchors=anchors, policy_key=CP.DEFAULT.key(identity['arm']),
                           robot=RM.robot_digest(), config_require=cfg.key())
    assert found is not None and Path(found.path).resolve() == path.resolve(), 'native exact lookup did not select this map'
    return {'artifact': ref(path), 'npz': ref(path.with_suffix('.npz')), 'source_motion': source,
            'q_start': lib['q_start'], 'positive_travel_cells': positive, 'verified': True, 'positive': positive > 0,
            'scope': 'Component only; each actual goal travel and scene must pass its own native queries.'}


def run(request, index):
    _, unit = verify(request, index)
    out = Path(unit['output'])
    out.mkdir(exist_ok=False)
    started = time.time()
    result = {'schema': 'isg-selected-artic-prefix-result-v1', 'campaign_id': 'campaign_20261110', 'unit_id': unit['id'],
              'request': ref(request), 'unit_request': unit['request'], 'owner': '/root', 'job_id': os.environ.get('SLURM_JOB_ID'),
              'started_at': started, 'cpu_affinity': sorted(os.sched_getaffinity(0)), 'status': 'running',
              'benchmark_credit': 0, 'production_authorization': False, 'artifacts': []}
    write(out / 'result.json', result)
    sys.path.insert(0, unit['native_runtime'])
    libraries = [p['path'] for p in unit['identity']['libraries']]
    cmd = [sys.executable, '-u', '-m', 'ispatialgym.sampling.reach_maps', 'prefix', '--artic-lib', *libraries,
           '--anchors', unit['anchors']['path'], '--heights', unit['identity']['height_key'], '--arm', unit['identity']['arm'],
           '--out', str(out / 'maps'), '--lattice', '.08', '--seeds', '8', '--batch', '2048']
    result['command'] = cmd
    log = out / 'native.log'
    try:
        with log.open('w') as f:
            try:
                process = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=unit['native_runtime'], timeout=3300)
                result['returncode'] = process.returncode
            except subprocess.TimeoutExpired:
                result['returncode'] = 'timeout_3300s'
        result['log'] = ref(log)
        result['native_done_marker'] = 'GB_PREFIX_DONE' in log.read_text(errors='replace')
        for path in sorted((out / 'maps').glob('prefix__*.json')):
            try:
                result['artifacts'].append(audit_map(path, unit))
            except Exception as exc:
                result['artifacts'].append({'artifact': ref(path), 'verified': False, 'error': repr(exc)})
            write(out / 'result.json', result)
        good = [p for p in result['artifacts'] if p.get('verified') and p.get('positive')]
        result['positive_maps'] = len(good)
        result['expected_maps'] = len(libraries)
        result['status'] = 'verified_prefix_components' if len(good) == len(libraries) else 'partial_or_failed_prefix_components'
        result['next_action'] = 'Join only exact verified positive motion/start/height components; diagnose each missing or zero-usable map; validate actual configuration queries.'
    except Exception as exc:
        result.update(status='failed_worker', error=repr(exc), next_action='Root diagnoses this exact worker failure; preserve computed maps for verification.')
    finally:
        result.update(finished_at=time.time(), elapsed_seconds=time.time()-started, consumers=unit['consumers'])
        write(out / 'result.json', result)
        print(json.dumps({k: v for k, v in result.items() if k not in ('consumers', 'artifacts', 'command')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--unit', type=int, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.verify_only:
        _, unit = verify(args.request, args.unit)
        print(json.dumps({'status': 'inputs_verified', 'unit': unit['id']}))
    else:
        run(args.request, args.unit)
