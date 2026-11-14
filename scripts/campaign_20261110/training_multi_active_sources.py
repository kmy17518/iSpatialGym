"""Exact source contracts for a bounded multi-active model-pool sequence."""
import copy
import re
from pathlib import Path
from .contracts import require, digest
from .training_pool_world import checked, active_actor_names

SCHEMA = 'isg-selected-multi-active-pool-episode-probe-v2'


def source_task_identity(source, record, config, bddl):
    """Keep the export task subtype distinct from the BDDL activity directory."""
    from .contracts import task_for_alias
    activity = bddl.parent.name
    definition_id = int(bddl.stem[7:])
    require(record['task'] == config['template']['task'] == task_for_alias(source['class_id'])
            and activity == record['task'].split('-', 1)[0]
            and source['problem'] == f'{activity}-{definition_id}',
            'source task subtype, BDDL activity or exact problem differs')
    return activity, definition_id


def source_contract(value, records):
    root = Path(value['campaign_root']).resolve()
    proof = checked(root, value['active_role_contract'])
    require(proof['schema'] == 'isg-two-selected-active-role-source-contract-v1'
            and proof['source_native_runtime'] == value['native_runtime']
            and proof['production_released'] is False and proof['benchmark_credit'] == 0
            and value['episodes'][0] == value['episodes'][2],
            'multi-active sequence must retain exact source runtime and repeated A')
    members = {row['member']: row for row in proof['members']}
    require(len(proof['members']) == len(members) == 2 and set(members) == {'A', 'B'},
            'multi-active source member inventory differs')
    world = checked(root, proof['common_world'])
    require(world['background_bytes_equal'] is True and world['support_source_equality_verified'] is True
            and world['new_obstacle_removals'] == [], 'multi-active shared world proof differs')
    checked(root, world['common_background'])
    tasks = []
    for label, episode, record in zip(value['sequence'], value['episodes'], records):
        source = members[label]; roles = source['expected']['active_roles']
        require(len(roles) > 1 and episode['expected']['active_roles'] == roles
                and episode['configuration'] == source['configuration']
                and record['class_id'] == source['class_id'] and record['variant'] == source['variant']
                and source['common_world_proof'] == proof['common_world'],
                'multi-active source role, configuration, class or variant differs')
        config = checked(root, episode['configuration'])
        setup = config['template']['scene_setup']
        require(setup['common_world_proof'] == proof['common_world'],
                'multi-active source task definition lacks exact configuration binding')
        from .training_pool_source_compatibility import checked_task_definition
        task_migration = checked_task_definition(root, source, setup)
        bddl = Path(source['bddl']['path']).resolve()
        from .contracts import file_digest
        require(bddl.is_relative_to(root) and file_digest(bddl) == source['bddl']['sha256']
                and re.fullmatch(r'problem\d+\.bddl', bddl.name) is not None,
                'multi-active exact source BDDL changed')
        activity, definition_id = source_task_identity(source, record, config, bddl)
        names = active_actor_names([row['native_name'] for row in roles.values()])
        require(len(names) == len(roles), 'multi-active source roles alias the same actor')
        scene = checked(root, record['artifacts']['recorded_scene'])
        bindings = scene['metadata']['task']['inst_to_name']
        for role, row in roles.items():
            require(bindings.get(row['instance']) == row['native_name'],
                    'recorded multi-active source role binding differs: ' + role)
        tasks.append({'activity': activity, 'definition_id': definition_id,
                      'bddl': copy.deepcopy(source['bddl']), 'variant': record['variant'],
                      'class_id': record['class_id'], 'role_map': copy.deepcopy(roles),
                      'source_task_metadata_sha256': digest(scene['metadata']['task'])})
        if task_migration is not None:
            tasks[-1]['fixed_support_task_migration'] = task_migration
        if 'fixed_support_role' in source:
            from .training_pool_source_compatibility import normalize_fixed_support
            normalize_fixed_support(scene, names, source['fixed_support_role'])
            tasks[-1]['fixed_support_role'] = copy.deepcopy(source['fixed_support_role'])
    require(set(tasks[0]['role_map']) == set(tasks[1]['role_map'])
            and set(row['native_name'] for row in tasks[0]['role_map'].values()) !=
                set(row['native_name'] for row in tasks[1]['role_map'].values()),
            'multi-active A/B roles or selected actor sets do not differ explicitly')
    return tasks
