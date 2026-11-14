"""Exact-source, zero-credit experiments on an explicitly changed common runtime.

This proves input/runtime identity, not physical equivalence or release. The
original bounded source authorization and all native comparisons still apply.
"""
import json
from pathlib import Path

from .contracts import file_digest, require

SCHEMA = 'isg-integrated-pipeline-candidate-v1'
REPLACED = {'integrated_candidate_v1', 'probe_runtime_sha256', 'runtime_compatibility'}


def reference(path):
    return {'path':str(path), 'sha256':file_digest(path)}


def checked(root, ref):
    path = Path(ref['path']).resolve()
    require(path.is_relative_to(root) and reference(path) == ref,
            'integrated candidate reference changed or escaped campaign')
    return path


def payloads(runtime):
    paths = [p for folder in ('ispatialgym','scripts') for p in (runtime/folder).rglob('*.py') if p.is_file()]
    require(paths and all(p.resolve().is_relative_to(runtime) for p in paths),
            'candidate Python payload inventory empty or escaped snapshot')
    return {str(p.relative_to(runtime)):file_digest(p) for p in paths}


def source_request(root, request):
    """Return the original operation settings after exact request projection."""
    root = Path(root).resolve()
    require(request.get('integrated_candidate_v1') is True
            and request.get('production_released') is False
            and type(request.get('benchmark_credit')) is int and request['benchmark_credit'] == 0
            and not request.get('physical_candidate'), 'integration requires bounded zero-credit scope')
    proof = json.loads(checked(root, request['runtime_compatibility']).read_text())
    require(proof.get('schema') == SCHEMA and proof.get('production_released') is False
            and proof.get('benchmark_credit') == 0
            and proof.get('physical_equivalence_established') is False
            and proof.get('source_inputs_changed') is False
            and proof.get('permitted_execution_kinds') == ['physics_action_open_loop'],
            'integration proof cannot imply equivalence or production release')
    base = json.loads(checked(root, proof['base_request']).read_text())
    require(not base.get('integrated_candidate_v1')
            and {k:v for k,v in request.items() if k not in REPLACED} ==
                {k:v for k,v in base.items() if k not in REPLACED},
            'integration changed source inputs, episode order, budgets or operation settings')
    pool = base.get('selected_pool_world') is True and base.get('pool_action_replay_v1') is True
    tray = base.get('late_incoming_physics_rebuild_v1') is True and base.get('fixed_mass_baseline') is True
    require(pool != tray and not base.get('load_only') and not base.get('render_cleanup_only')
            and not base.get('sequential_render_only'), 'unmeasured integration path')
    return base, proof


def verify(root, request, records, runtime, verify_base):
    root = Path(root).resolve(); runtime = Path(runtime).resolve()
    base, proof = source_request(root, request)
    baseline = checked(root, proof['base_runtime']).parent
    require(baseline.is_relative_to(root/'code') and runtime.is_relative_to(root/'code')
            and base['probe_runtime_sha256'] == proof['base_runtime']['sha256']
            and reference(runtime/'.snapshot_commit') == proof['candidate_runtime']
            and request['probe_runtime_sha256'] == proof['candidate_runtime']['sha256'],
            'integrated execution or original runtime differs')
    # Preserve every original source and isolated-runtime check. This callback
    # sees the original request, so cannot recurse through integration.
    require(verify_base(root, base, records, baseline) == base['runtime_compatibility'],
            'original isolated runtime proof failed')
    actual = payloads(runtime); previous = payloads(baseline)
    require(actual == proof['candidate_python_payloads'], 'integrated candidate payload changed')
    differences = {name:{'base':previous.get(name),'candidate':actual.get(name)}
                   for name in set(previous)|set(actual) if previous.get(name) != actual.get(name)}
    require(differences == proof['changed_payloads'], 'integrated candidate changes were not fully declared')
    return request['runtime_compatibility']


def renderer_source_request(root, request):
    root = Path(root).resolve()
    require(request.get('integrated_renderer_candidate_v1') is True
            and request.get('production_released') is False
            and type(request.get('benchmark_credit')) is int and request['benchmark_credit'] == 0
            and 'changed_payloads' not in request, 'integrated renderer scope or payload declaration ambiguous')
    proof = json.loads(checked(root, request['integration_proof']).read_text())
    require(proof.get('schema') == SCHEMA and proof.get('production_released') is False
            and proof.get('benchmark_credit') == 0 and proof.get('physical_equivalence_established') is False
            and proof.get('source_inputs_changed') is False
            and proof.get('permitted_execution_kinds') == ['kinematic_observation_rendering'],
            'renderer integration cannot claim physics equivalence or release')
    base = json.loads(checked(root, proof['base_request']).read_text())
    replaced = {'candidate_runtime','changed_payloads','integrated_renderer_candidate_v1','integration_proof'}
    require(not base.get('integrated_renderer_candidate_v1')
            and base.get('schema') == 'isg-pool-native-render-reuse-request-v1'
            and {k:v for k,v in request.items() if k not in replaced} ==
                {k:v for k,v in base.items() if k not in replaced},
            'integrated renderer changed recording selection, frames, cleanup or physical scope')
    return base, proof


def verify_renderer(root, request, runtime, validate_base):
    root = Path(root).resolve(); runtime = Path(runtime).resolve()
    base, proof = renderer_source_request(root, request)
    baseline = checked(root, proof['base_runtime']).parent
    require(baseline.is_relative_to(root/'code') and runtime.is_relative_to(root/'code')
            and base['candidate_runtime'] == proof['base_runtime']
            and request['candidate_runtime'] == proof['candidate_runtime'] == reference(runtime/'.snapshot_commit'),
            'renderer integrated or original runtime differs')
    original = validate_base(root, proof['base_request']['path'], runtime=baseline)
    require(original[0] == base, 'original renderer input checks failed')
    actual = payloads(runtime); previous = payloads(baseline)
    require(actual == proof['candidate_python_payloads'], 'integrated renderer payload changed')
    differences = {name:{'base':previous.get(name),'candidate':actual.get(name)}
                   for name in set(previous)|set(actual) if previous.get(name) != actual.get(name)}
    require(differences == proof['changed_payloads'], 'integrated renderer changes not fully declared')
    return proof


def current_recording_renderer_source(root, request):
    """Bind newly verified recordings without rewriting their generation identity."""
    root = Path(root).resolve()
    require(request.get('current_recording_renderer_v1') is True
            and request.get('production_released') is False
            and type(request.get('benchmark_credit')) is int and request['benchmark_credit'] == 0
            and not request.get('integrated_renderer_candidate_v1')
            and 'changed_payloads' not in request and 'integration_proof' not in request
            and request.get('runtime_compatibility') == request.get('current_recording_proof'),
            'new-recording renderer scope is ambiguous')
    proof = json.loads(checked(root, request['current_recording_proof']).read_text())
    require(proof.get('schema') == 'isg-current-recording-renderer-proof-v1'
            and proof.get('production_released') is False and proof.get('benchmark_credit') == 0
            and proof.get('permitted_execution_kinds') == ['kinematic_observation_rendering'],
            'new recordings do not authorize physical execution or release')
    base = json.loads(checked(root, proof['base_request']).read_text())
    require(base.get('integrated_renderer_candidate_v1') is True
            and not base.get('current_recording_renderer_v1'), 'renderer base must be the checked common candidate')
    changed = {'candidate_runtime','parent_runtime','cold_driver','cold_requests','cold_verification',
               'scene_union_sha256','integrated_renderer_candidate_v1','integration_proof',
               'current_recording_renderer_v1','current_recording_proof','runtime_compatibility'}
    require({k:v for k,v in request.items() if k not in changed} ==
            {k:v for k,v in base.items() if k not in changed},
            'new-recording renderer changed sequence, physics, cleanup, or source configuration')
    physics = json.loads(checked(root, proof['source_recording_verification']).read_text())
    require(physics.get('status') == 'matched' and physics.get('active_recording_scope_verified') is True
            and physics.get('manifest') == proof['source_recording_manifest'],
            'new recordings lack complete active-only native action verification')
    checked(root, proof['source_recording_manifest'])
    cold = json.loads(checked(root, request['cold_verification']).read_text())
    require(cold.get('status') == 'native_cold_render_verified'
            and set(cold['controls']) == {'FRESH_A','FRESH_B'}
            and all(v.get('all_original_cold_frames_matched') is True and v.get('export_verification')
                    for v in cold['controls'].values()), 'new recordings lack full original-cold frame/export proof')
    for control, reference in zip(('FRESH_A','FRESH_B'),request['cold_requests']):
        value = json.loads(checked(root, reference).read_text())
        require(value['control'] == control
                and value['recording_manifest'] == proof['source_recording_manifest']
                and value['active_recording_verification'] == proof['source_recording_verification']
                and value['driver'] == request['cold_driver']
                and value['native_runtime'] == request['parent_runtime'],
                'cold native renderer is not bound to these newly verified recordings')
    require(len(request['cold_requests']) == 2, 'two exact new-recording cold references required')
    return base, proof


def verify_current_recording_renderer(root, request, runtime, validate_base):
    root = Path(root).resolve(); runtime = Path(runtime).resolve()
    base, proof = current_recording_renderer_source(root, request)
    baseline = checked(root, proof['base_runtime']).parent
    require(baseline.is_relative_to(root/'code') and runtime.is_relative_to(root/'code')
            and base['candidate_runtime'] == proof['base_runtime']
            and reference(runtime/'.snapshot_commit') == request['candidate_runtime'] == proof['candidate_runtime'],
            'new-recording native runtime differs')
    require(validate_base(root, proof['base_request']['path'], runtime=baseline)[0] == base,
            'original common renderer checks failed')
    before = payloads(baseline); after = payloads(runtime)
    changes = {n:{'base':before.get(n),'candidate':after.get(n)}
               for n in set(before)|set(after) if before.get(n) != after.get(n)}
    require(after == proof['candidate_python_payloads'] and changes == proof['changed_payloads'],
            'new-recording candidate has undeclared payloads')
    return proof
