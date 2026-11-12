"""Build exact measured-height articulation prefixes and verify returned maps."""
import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
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


def verify_ref(value, sources=None):
    sources = sources or {}
    assert ref(sources.get(value['path'], value['path']))['sha256'] == value['sha256'], value['path']
    if value.get('relocation_proof'):
        verify_ref(value['relocation_proof'], sources)


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def load_input_relocations(path, manifest=None, expected_sha256=None):
    if manifest is None:
        assert expected_sha256 is None, 'relocation digest without manifest'
        return {}, None
    assert expected_sha256 and ref(manifest)['sha256'] == expected_sha256, 'input relocation manifest digest differs'
    mapping = read(manifest)
    assert mapping['schema'] == 'isg-selected-artic-prefix-input-relocations-v1'
    assert mapping['campaign_id'] == 'campaign_20261110' and mapping['source_request'] == ref(path)
    wave = Path(path).resolve().parent.name
    allowed = [c / 'training_preparation' / wave for c in CAMPAIGN_ROOTS]
    sources = {}
    for original, entry in mapping['files'].items():
        destination = Path(entry['path'])
        assert destination.is_absolute() and '..' not in destination.parts
        assert any(destination.is_relative_to(p) and destination.resolve().is_relative_to(p.resolve()) for p in allowed), 'input copy outside isolated wave'
        assert ref(destination)['sha256'] == entry['sha256'], 'input copy changed'
        sources[original] = destination
    return sources, ref(manifest)


def resolve_output(path, wave, unit, output_root=None):
    """Relocate output only, preserving the immutable wave and native unit."""
    original = Path(unit['output'])
    if output_root is None:
        return original, None
    source = Path(path).resolve()
    root = Path(output_root)
    assert root.is_absolute() and '..' not in root.parts, 'output root must be an absolute campaign path'
    assert re.fullmatch(r'prefix_[0-9a-f]{24}', unit['id']), 'invalid physical unit identity'
    # The caller may relocate only into this same named preparation wave.
    assert source.parent.name.startswith('selected_artic_prefix_'), 'foreign request scope'
    assert any(source.is_relative_to(c / 'training_preparation') for c in CAMPAIGN_ROOTS), 'foreign request root'
    allowed = [c / 'training_preparation' / source.parent.name for c in CAMPAIGN_ROOTS]
    resolved = root.resolve()
    assert any(root.is_relative_to(p) and resolved.is_relative_to(p.resolve()) for p in allowed), 'output root outside isolated campaign wave'
    assert original.resolve().is_relative_to(source.parent), 'original output outside request wave'
    out = resolved / unit['id']
    assert out.resolve().is_relative_to(resolved), 'unit output escaped relocated root'
    mapping = {'schema': 'isg-selected-artic-prefix-output-relocation-v1', 'campaign_id': wave['campaign_id'],
               'request': ref(path), 'unit_request': unit['request'], 'unit_id': unit['id'], 'unit_index': unit['index'],
               'unit_payload_sha256': hashlib.sha256(json.dumps(unit, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
               'original_output': str(original), 'output_root': str(resolved), 'actual_output': str(out),
               'native_runtime': unit['native_runtime'], 'native_inputs_unchanged': True,
               'worker': ref(__file__), 'benchmark_credit': 0, 'production_authorization': False}
    return out, mapping


def verify(request, index, input_relocations=None, input_relocations_sha256=None):
    sources, _ = load_input_relocations(request, input_relocations, input_relocations_sha256)
    wave = read(request)
    assert wave['schema'] == 'isg-selected-artic-prefix-wave-v1' and wave['campaign_id'] == 'campaign_20261110'
    assert wave['benchmark_credit'] == 0 and wave['production_authorization'] is False
    unit = wave['units'][index]
    assert unit['index'] == index and unit['native_runtime'] == wave['native_runtime']
    for proof in [wave['selected_manifest'], wave['motion_wave'], unit['request'], unit['anchors'], unit['motion_request'], unit['motion_result']]:
        verify_ref(proof, sources)
    for proof in unit['identity']['libraries']:
        verify_ref(proof, sources)
    for consumer in unit['consumers']:
        verify_ref(consumer['support_artifact'], sources)
        assert abs(consumer['measured_height'] - float(unit['identity']['height_key'])) <= .00005
    for name, sha in unit['identity']['generator_files'].items():
        verify_ref({'path': str(Path(unit['native_runtime']) / name), 'sha256': sha})
    return wave, unit


def audit_map(path, unit, sources=None):
    sources = sources or {}
    import numpy as np
    from ispatialgym.grasping import schema as S, contact_policy as CP
    from ispatialgym.sampling import reach_maps as RM, height_anchors as HA, reach_bands as RB

    record = RM.PrefixMap(str(path))
    meta = record.meta
    identity = unit['identity']
    source = next((r for r in identity['libraries'] if str(sources.get(r['path'], r['path'])) == meta.get('source', {}).get('library')), None)
    assert source is not None, 'map names a different source motion'
    verify_ref(source, sources)
    lib = read(sources.get(source['path'], source['path']))
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
    anchors = HA.AnchorSet(read(sources.get(unit['anchors']['path'], unit['anchors']['path']))['anchors'], max_approx_m=0)
    # The native lookup reads JSON metadata, so tuple-valued config fields
    # must use their serialized list representation for exact comparison.
    found = RM.find_prefix(str(path.parent), spec, anchors=anchors, policy_key=CP.DEFAULT.key(identity['arm']),
                           robot=RM.robot_digest(), config_require=json.loads(json.dumps(cfg.key())))
    assert found is not None and Path(found.path).resolve() == path.resolve(), 'native exact lookup did not select this map'
    return {'artifact': ref(path), 'npz': ref(path.with_suffix('.npz')), 'source_motion': source, 'resolved_source_motion': ref(sources.get(source['path'], source['path'])),
            'q_start': lib['q_start'], 'positive_travel_cells': positive, 'verified': True, 'positive': positive > 0,
            'scope': 'Component only; each actual goal travel and scene must pass its own native queries.'}


def run(request, index, output_root=None, input_relocations=None, input_relocations_sha256=None):
    wave, unit = verify(request, index, input_relocations, input_relocations_sha256)
    sources, input_mapping = load_input_relocations(request, input_relocations, input_relocations_sha256)
    out, relocation = resolve_output(request, wave, unit, output_root)
    if relocation is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)
    if relocation is not None:
        relocation['input_relocations'] = input_mapping
        write(out / 'output_relocation.json', relocation)
    started = time.time()
    result = {'schema': 'isg-selected-artic-prefix-result-v1', 'campaign_id': 'campaign_20261110', 'unit_id': unit['id'],
              'request': ref(request), 'unit_request': unit['request'], 'owner': '/root', 'job_id': os.environ.get('SLURM_JOB_ID'),
              'started_at': started, 'cpu_affinity': sorted(os.sched_getaffinity(0)), 'status': 'running',
              'benchmark_credit': 0, 'production_authorization': False, 'artifacts': []}
    result.update(actual_output=str(out), original_output=unit['output'], input_relocations=input_mapping)
    if relocation is not None:
        result['output_relocation'] = ref(out / 'output_relocation.json')
    write(out / 'result.json', result)
    sys.path.insert(0, unit['native_runtime'])
    libraries = [str(sources.get(p['path'], p['path'])) for p in unit['identity']['libraries']]
    cmd = [sys.executable, '-u', '-m', 'ispatialgym.sampling.reach_maps', 'prefix', '--artic-lib', *libraries,
           '--anchors', str(sources.get(unit['anchors']['path'], unit['anchors']['path'])), '--heights', unit['identity']['height_key'], '--arm', unit['identity']['arm'],
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
                result['artifacts'].append(audit_map(path, unit, sources))
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
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--input-relocations', type=Path)
    parser.add_argument('--input-relocations-sha256')
    args = parser.parse_args()
    if args.verify_only:
        wave, unit = verify(args.request, args.unit, args.input_relocations, args.input_relocations_sha256)
        out, relocation = resolve_output(args.request, wave, unit, args.output_root)
        print(json.dumps({'status': 'inputs_verified', 'unit': unit['id'], 'actual_output': str(out),
                          'output_relocation': relocation, 'input_relocations': ref(args.input_relocations) if args.input_relocations else None}))
    else:
        run(args.request, args.unit, args.output_root, args.input_relocations, args.input_relocations_sha256)
