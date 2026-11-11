"""Replace an unmaterialized support selection with exact native default evidence.

Selected slot identities remain stable. Changed logical requests get new hashes;
existing physical proofs and materialized configurations cannot be reinterpreted.
"""
from copy import deepcopy

from scripts.campaign_20261110.training_dependency_inventory import digest


def replace_support(manifest, *, old_profile, new_profile, binding, correction_ref):
    out = deepcopy(manifest)
    old_id, new_id = old_profile['id'], new_profile['id']
    if old_id == new_id or old_profile['sampled_uniform_factors'] != new_profile['sampled_uniform_factors']:
        raise ValueError('fresh identity must preserve the exact selected factors')
    old_scales, new_scales = old_profile['absolute_scales'], new_profile['absolute_scales']
    model_key = new_profile['category'] + '/' + new_profile['model']

    def transform(value):
        if isinstance(value, list):
            return [transform(v) for v in value]
        if not isinstance(value, dict):
            return value
        result = {k: transform(v) for k, v in value.items()}
        if (value.get('support_profile') == old_id and
                value.get('scene') == binding['scene'] and value.get('object') == binding['name']):
            slot = value['height_slot']
            if slot not in (0, 1) or value['selected_scale'] != old_scales[slot]:
                raise ValueError('selected support slot differs from original profile')
            result.update(support_profile=new_id, selected_scale=new_scales[slot])
        if (value.get('role') == 'support' and value.get('model_key') == model_key and
                value.get('scale_rule') == 'z_only_support_profile'):
            if value.get('asset_sha256') != new_profile['asset_sha256']:
                raise ValueError('support asset changed')
            try:
                slot = old_scales.index(value['applied_scale'])
            except ValueError as exc:
                raise ValueError('support role is not one of the original selected scales') from exc
            result.update(applied_scale=new_scales[slot],
                          original_default_scale=new_profile['default_scale'], scale_profile=new_id)
        return result

    replacements = {}
    for request in out['artifact_requests']:
        inputs = transform(request['inputs'])
        if inputs == request['inputs']:
            continue
        if request['status'] in ('verified', 'not_applicable'):
            raise ValueError('cannot reinterpret an already verified dependency')
        old = request['id']
        new = 'request_' + digest({'kind': request['kind'], 'inputs': inputs})
        if new == old:
            raise ValueError('changed inputs did not receive a fresh identity')
        replacements[old] = new
        request.update(id=new, inputs=inputs, physical_artifact_id=None, artifact=None,
                       evidence=[], resolved_inputs={}, active_jobs=[],
                       selection_correction={'previous_request_id': old, 'evidence': correction_ref})
    if not replacements:
        raise ValueError('no exact original support requests found')
    ids = [r['id'] for r in out['artifact_requests']]
    if len(set(ids)) != len(ids):
        raise ValueError('replacement request identity collision')
    changed_configs = []
    for config in out['configurations']:
        deps = [replacements.get(k, k) for k in config['dependencies']]
        roles = transform(config['roles'])
        if deps == config['dependencies'] and roles == config['roles']:
            continue
        if config['configuration_sha256'] is not None:
            raise ValueError('cannot change a materialized configuration')
        config.update(dependencies=deps, roles=roles,
                      selection_correction={'evidence': correction_ref,
                                            'selected_slots_preserved': True})
        changed_configs.append(config['id'])
    return out, {'request_replacements': replacements, 'configuration_ids': changed_configs,
                 'selected_slots_preserved': True, 'materialized_configurations_changed': 0,
                 'physical_artifacts_reinterpreted': 0, 'evidence': correction_ref}
