"""Recompile only an exact source-bound incoming task in a loaded pool world."""
import copy
from pathlib import Path
from .contracts import require, file_digest, digest

CONTEXT_KEYS = frozenset({'ISPATIALGYM_TRAINING_OBSERVATION_REQUEST',
    'ISPATIALGYM_TRAINING_OBSERVATION_REQUEST_SHA256',
    'ISPATIALGYM_CAMPAIGN_RESTORATION_PROBE', 'ISPATIALGYM_CAMPAIGN_RESTORATION_CONTEXT'})


def select_episode_environment(environment, source, delta):
    """Replace source task settings while retaining this exact observer context."""
    require(all(isinstance(k, str) and k.startswith(('ISPATIALGYM_', 'ISG_'))
                and isinstance(v, str) for values in (source, delta) for k, v in values.items()),
            'pool source execution settings must be explicit string task flags')
    context = {key: environment[key] for key in CONTEXT_KEYS if key in environment}
    require({'ISPATIALGYM_TRAINING_OBSERVATION_REQUEST',
             'ISPATIALGYM_TRAINING_OBSERVATION_REQUEST_SHA256'} <= context.keys(),
            'multi-active task switch lacks its current observation context')
    prior = {key: value for key, value in environment.items() if key.startswith(('ISPATIALGYM_', 'ISG_'))}
    for key in prior:
        del environment[key]
    environment.update(source); environment.update(delta); environment.update(context)
    actual = {key: value for key, value in environment.items() if key.startswith(('ISPATIALGYM_', 'ISG_'))}
    return {'source_environment_sha256': digest(source), 'declared_delta': dict(delta),
            'observation_context': context, 'effective_task_environment_sha256': digest(actual),
            'removed_prior_task_keys': sorted(set(prior) - set(actual))}


def verify_definition(loader, member):
    spec = member['task_definition']; path = Path(spec['bddl']['path']).resolve()
    require(path.is_file() and file_digest(path) == spec['bddl']['sha256']
            and path.name == 'problem' + str(spec['definition_id']) + '.bddl'
            and path.parent.name == spec['activity']
            and loader.activity == spec['activity'] and loader.did == spec['definition_id']
            and loader.tro.meta.task_variant == spec['variant'],
            'incoming pool task does not match its exact source BDDL and variant')
    require(dict(loader.tro.meta.object_bindings) == member['bindings'],
            'incoming pool task bindings differ from exact source')
    return spec, path


def observe_task(loader, member):
    spec, path = verify_definition(loader, member); task = loader.task
    require(task.activity_name == spec['activity'] and task.activity_definition_id == spec['definition_id']
            and task.task_variant == spec['variant'] and task._tro is not None
            and task._tro.meta.task_variant == spec['variant']
            and dict(task._tro.meta.object_bindings) == member['bindings'],
            'compiled native task retains a different source problem or variant')
    bindings = {key: value['scene_object_name'] for key, value in member['bindings'].items()}
    for role, row in spec['role_map'].items():
        entity = task.object_scope.get(row['instance'])
        require(entity is not None and entity.name == row['native_name']
                and bindings[row['instance']] == row['native_name'],
                'compiled native task lacks exact active role: ' + role)
    return {'activity': task.activity_name, 'definition_id': task.activity_definition_id,
            'variant': task.task_variant, 'class_id': spec['class_id'], 'bddl': spec['bddl'],
            'active_roles': copy.deepcopy(spec['role_map']), 'bindings_sha256': digest(member['bindings']),
            'source_task_metadata_sha256': spec['source_task_metadata_sha256'],
            'physics_steps_requested': 0,
            **({'fixed_support_task_migration':copy.deepcopy(spec['fixed_support_task_migration'])}
               if spec.get('fixed_support_task_migration') is not None else {})}


def rebind_task(loader, member):
    spec, path = verify_definition(loader, member)
    task = loader.task
    task.activity_definitions_root = str(path.parent.parent)
    task._tro = loader.tro
    task._config_object_bindings = copy.deepcopy(member['bindings'])
    task.update_activity(loader.env, spec['activity'], spec['definition_id'])
    ok, feedback = task.initialize_activity(loader.env)
    require(ok, 'incoming exact pool task initialization failed: ' + str(feedback))
    loader.cfg = copy.deepcopy(loader.cfg)
    loader.cfg['task'].update(activity_name=spec['activity'], activity_definition_id=spec['definition_id'],
                              activity_definitions_root=str(path.parent.parent), tro_path=loader.tro_path)
    return observe_task(loader, member)
