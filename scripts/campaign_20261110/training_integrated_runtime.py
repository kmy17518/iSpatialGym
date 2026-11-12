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
