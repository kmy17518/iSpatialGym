"""Reviewed source projections for unchanged physical model-pool worlds.

These checks preserve each recorded source. They authorize neither execution
nor a source rewrite, and do not imply native one-world equivalence.
"""
import copy
import re
from pathlib import Path
from .contracts import require, digest, file_digest


def checked_task_definition(root, source, setup):
    """Permit only an explicit fixed-support token substitution in new inputs."""
    from .training_pool_world import checked
    root = Path(root).resolve()
    runtime_ref = source['bddl']
    path = Path(runtime_ref['path']).resolve()
    require(path.is_relative_to(root) and file_digest(path) == runtime_ref['sha256']
            and setup['runtime_bddl_sha256'] == runtime_ref['sha256'],
            'exact runtime task definition changed')
    original_ref = source.get('original_bddl')
    migration_ref = source.get('fixed_support_bddl_substitution')
    require((original_ref is None) == (migration_ref is None),
            'support task substitution needs original and exact migration proof')
    if migration_ref is None:
        require(setup['source_bddl_sha256'] == runtime_ref['sha256'],
                'task definition changed without reviewed support substitution')
        return None
    migration = checked(root, migration_ref)
    require(migration.get('schema') == 'isg-fixed-support-bddl-substitution-v1'
            and migration.get('production_released') is False
            and migration.get('benchmark_credit') == 0
            and migration['source_bddl'] == original_ref
            and migration['runtime_bddl'] == runtime_ref
            and migration['common_world'] == source['common_world_proof'] == setup['common_world_proof']
            and setup['source_bddl_sha256'] == original_ref['sha256'],
            'support task substitution source, world or runtime binding differs')
    original = Path(original_ref['path']).resolve()
    require(original.is_relative_to(root) and file_digest(original) == original_ref['sha256'],
            'original support task definition changed')
    old, new = migration['source_synset'], migration['runtime_synset']
    old_instance, new_instance = migration['source_instance'], migration['runtime_instance']
    require(all(isinstance(token, str) and re.fullmatch(r'[a-z_]+\.n\.\d+', token)
                for token in (old, new)) and old != new
            and old_instance == old + '_1' and new_instance == new + '_1'
            and not {old_instance, new_instance}.intersection(
                row['instance'] for row in source['expected']['active_roles'].values()),
            'support task substitution cannot rename an active role or arbitrary token')
    mapping = {old_instance:new_instance, old:new}
    pattern = r'(?<![A-Za-z0-9_.-])(?:' + '|'.join(map(re.escape, mapping)) + r')(?![A-Za-z0-9_.-])'
    before, after = original.read_bytes(), path.read_bytes()
    text = before.decode('utf-8'); counts = {token:0 for token in mapping}
    def replace(match):
        counts[match[0]] += 1
        return mapping[match[0]]
    changed = re.sub(pattern, replace, text).encode('utf-8')
    require(all(counts.values()) and changed == after,
            'runtime BDDL changed beyond the exact declared fixed support tokens')
    pool = checked(root, setup['native_model_pool'])
    require(pool['fixed_bindings']['support'] == [migration['native_support']],
            'task support substitution changes the actual fixed support')
    return {'original_bddl':copy.deepcopy(original_ref),
            'fixed_support_bddl_substitution':copy.deepcopy(migration_ref),
            'exact_token_counts':counts}


def normalize_fixed_support(document, active, role):
    """Normalize one declared BDDL support key, retaining its physical actor."""
    require(isinstance(role, dict) and set(role) == {
        'instance', 'native_name', 'initialized_sha256', 'state_sha256'},
        'fixed support needs its exact source binding and physical hashes')
    instance, name = role['instance'], role['native_name']
    require(isinstance(instance, str) and instance and isinstance(name, str) and name
            and name not in active and not name.startswith('campaign_pool_'),
            'fixed support cannot be a selected pool actor')
    initial = document['objects_info']['init_info']
    states = document['state']['registry']['object_registry']
    bindings = document['metadata']['task']['inst_to_name']
    require(name in initial and name in states
            and digest(initial[name]) == role['initialized_sha256']
            and digest(states[name]) == role['state_sha256']
            and {key for key, value in bindings.items() if value == name} == {instance},
            'fixed support source geometry, state or binding differs')
    result = copy.deepcopy(document)
    bindings = result['metadata']['task']['inst_to_name']
    token = '<fixed_pool_support>'
    require(token not in bindings, 'fixed support normalization collides')
    del bindings[instance]
    bindings[token] = name
    return result


def additive_manifest_union(root, proof, references):
    """Check each original manifest and an exact additive common manifest."""
    from .training_pool_world import checked
    from ispatialgym.sampling.model_pool import read_manifest
    reference = proof['pool_manifest_compatibility']
    declaration = checked(root, reference)
    unique = {digest(item): item for item in references}
    declared = declaration.get('source_manifests', [])
    require(declaration.get('schema') == 'isg-additive-pool-manifest-compatibility-v1'
            and declaration.get('production_released') is False
            and declaration.get('benchmark_credit') == 0
            and len(declared) == len(unique)
            and {digest(item): item for item in declared} == unique,
            'additive pool proof does not bind every exact source manifest')
    union_ref = declaration['union_manifest']
    union = checked(root, union_ref)
    require(union_ref in references, 'additive manifest must be an actual source manifest')
    for item in unique.values():
        source = checked(root, item)
        read_manifest(item['path'], scene=source['scene'], rooms=source['rooms'],
                      fixed_bindings=source['fixed_bindings'])
        require(all(source[key] == union[key] for key in (
            'scene', 'rooms', 'fixed_bindings', 'context_assets', 'reservation_policy_sha256')),
            'additive pool changes fixed world or reservation scope')
        require(source['entries'] and all(row in union['entries'] for row in source['entries']),
                'additive pool changed or removed an existing physical slot')
    return union_ref, reference
