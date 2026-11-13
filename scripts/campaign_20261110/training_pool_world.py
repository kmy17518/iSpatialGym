"""Exact scene contracts for bounded one-world model/scale reuse experiments.

Actors retain their initialized model and scale. Only selection, dynamic state
and isolation change between episodes. This module grants no release.
"""
import copy
import json
import math
from pathlib import Path
from .contracts import require, file_digest, digest
from ispatialgym.oracle.campaign_batch import compare_saved_scenes

SCHEMA='isg-model-pool-world-contract-v1'


def active_actor_names(value):
    values = [value] if isinstance(value, str) else value
    require(isinstance(values, (list, tuple, set, frozenset)) and values
            and all(isinstance(name, str) and name for name in values)
            and len(values) == len(set(values)), 'explicit unique active actor set required')
    return tuple(sorted(values))


def private_member_filename(value):
    names = active_actor_names(value)
    key = names[0] if len(names) == 1 else 'actors_' + digest(list(names))
    require('/' not in key and '\\' not in key, 'pool actor name cannot be a path')
    return 'private_union_' + key + '.json'


def isolated_member_scene(contract, document, active):
    """Build one explicit union initialization without changing a source scene.

    Inactive actors retain their exact initialized model/scale and serialized
    articulation state. Their roots start in separate out-of-room parking
    cells. Native collision/gravity/visibility isolation is a separate required
    operation before any episode restore; a parked JSON pose does not prove it.
    """
    active = set(active_actor_names(active))
    require(contract.get('schema') == SCHEMA and active <= set(contract['initialized_actors']),
            'unknown active pool member')
    names = set(contract['initialized_actors'])
    initial = document['objects_info']['init_info']
    states = document['state']['registry']['object_registry']
    require({n for n in initial if n.startswith('campaign_pool_')} == active
            and set(initial) == set(states)
            and all(initial[name] == contract['initialized_actors'][name] for name in active),
            'member scene identity differs from checked pool contract')
    result = copy.deepcopy(document)
    # Keep construction order identical in independent A/B worlds. Starting
    # with each source's active actor would otherwise reverse native pool IDs.
    result['objects_info']['init_info'] = {
        **{name: value for name, value in result['objects_info']['init_info'].items() if name not in names},
        **{name: copy.deepcopy(contract['initialized_actors'][name]) for name in sorted(names)}}
    destinations = result['state']['registry']['object_registry']
    isolation = {}
    for ordinal, name in enumerate(sorted(names)):
        flags = copy.deepcopy(contract['active_properties'][name])
        if name not in active:
            state = copy.deepcopy(contract['source_actor_states'][name])
            root = state.get('root_link', {})
            require(all(k in root for k in ('pos', 'ori', 'lin_vel', 'ang_vel')),
                    'inactive actor lacks a complete native root state')
            root.update(pos=[40. + ordinal, 40., 5.], lin_vel=[0., 0., 0.], ang_vel=[0., 0., 0.])
            destinations[name] = state
            flags['visible'] = False
            for link in flags['links'].values():
                if link['gravity_disabled'] is not None:
                    link['gravity_disabled'] = True
                link['collisions'] = {shape: False for shape in link['collisions']}
        isolation[name] = {'active': name in active, 'properties': flags,
                           'state': copy.deepcopy(destinations[name])}
    require(set(destinations) == set(result['objects_info']['init_info']),
            'union scene object/state inventory differs')
    result['state']['registry']['object_registry'] = {
        name: destinations[name] for name in result['objects_info']['init_info']}
    return result, isolation


def verify_pool_isolation(expected, observed, *, task_bindings, planner_names):
    """Require native properties and actual planner exclusion for every actor."""
    require(expected and expected.keys() == observed.keys(), 'native pool inventory differs')
    active = {name for name, row in expected.items() if row['active']}
    require(active and active <= set(task_bindings.values()),
            'selected actor lacks its exact task binding')
    inactive = expected.keys() - active
    require(not inactive.intersection(task_bindings.values()), 'inactive actor is task-bound')
    for name, row in expected.items():
        actual = observed[name]
        require(actual['properties'] == row['properties'],
                'native pool physical properties differ: '+name)
        if not row['active']:
            require((planner_names is None or name not in planner_names) and not actual.get('assisted_held')
                    and not actual.get('attached_parent')
                    and math.dist(actual['position'], row['state']['root_link']['pos']) <= 1e-6
                    and all(abs(float(x)) <= 1e-7 for k in ('linear_velocity', 'angular_velocity')
                            for x in actual[k]),
                    'inactive actor remains physically or logically active: '+name)
    return {'active': sorted(active), 'inactive': sorted(inactive),
            'properties_verified': True, 'planner_exclusion_verified': planner_names is not None,
            'scope': 'Native reads at this boundary only; export and per-step persistence require separate evidence.'}


def checked(root, reference):
    path=Path(reference['path']).resolve()
    require(path.is_relative_to(Path(root).resolve()) and file_digest(path)==reference['sha256'],
            'pool world source changed or escaped campaign')
    return json.loads(path.read_text())


def cold_anchor_transform(first_spawn, first_anchor, incoming_spawn):
    """Carry the measured cold anchor offset through the new spawn's SE(3)."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    for value in (first_spawn,first_anchor,incoming_spawn):
        require(set(value)>={'position','orientation'} and len(value['position'])==3
                and len(value['orientation'])==4 and
                all(math.isfinite(float(x)) for k in ('position','orientation') for x in value[k])
                and np.linalg.norm(value['orientation'])>0,'invalid cold anchor pose')
    a=Rotation.from_quat(first_spawn['orientation']);b=Rotation.from_quat(incoming_spawn['orientation'])
    delta=b*a.inv()
    position=np.asarray(incoming_spawn['position'])+delta.apply(
        np.asarray(first_anchor['position'])-np.asarray(first_spawn['position']))
    orientation=(delta*Rotation.from_quat(first_anchor['orientation'])).as_quat()
    return {'position':position.tolist(),'orientation':orientation.tolist()}


def normalize_active_role_bindings(document, active, roles):
    """Compare semantic roles without discarding any other source task metadata."""
    require(isinstance(roles, dict) and len(roles) == len(active)
            and all(isinstance(role, str) and role for role in roles),
            'explicit role for every active actor required')
    names = [row.get('native_name') for row in roles.values()]
    instances = [row.get('instance') for row in roles.values()]
    require(set(names) == set(active) and len(set(names)) == len(names)
            and all(isinstance(instance, str) and instance for instance in instances)
            and len(set(instances)) == len(instances), 'active role identities are not bijective')
    result = copy.deepcopy(document)
    bindings = result['metadata']['task']['inst_to_name']
    require({key for key, name in bindings.items() if name in active} == set(instances),
            'source active task binding is missing, duplicated or undeclared')
    normalized = {}
    for role, row in roles.items():
        instance, name = row['instance'], row['native_name']
        token = '<selected_pool_role:' + role + '>'
        require(bindings.get(instance) == name and token not in bindings,
                'source role binding differs or normalization collides')
        normalized[token] = token
        del bindings[instance]
    bindings.update(normalized)
    return result


def scene_contract(documents, active_names, pool_reference, *, role_maps=None,
                   source_pool_references=None, fixed_support_roles=None):
    """Verify shared static geometry, allowing only explicitly selected actors.

    Source scenes remain immutable. The returned union is a new, declared
    experimental initialization artifact, never substituted for source exports.
    """
    require(len(documents)==len(active_names)>=2,'pool contract needs distinct member scenes')
    active_sets = [active_actor_names(value) for value in active_names]
    if role_maps is not None:
        require(len(role_maps) == len(documents) and all(isinstance(roles, dict) for roles in role_maps)
                and all(set(roles) == set(role_maps[0]) for roles in role_maps),
                'source semantic role inventories differ')
    references = source_pool_references or [pool_reference] * len(documents)
    require(len(references) == len(documents)
            and (source_pool_references is None or role_maps is not None),
            'different source manifests require explicit multi-active roles')
    if fixed_support_roles is not None:
        require(role_maps is not None and len(fixed_support_roles) == len(documents)
                and all(isinstance(role, dict) for role in fixed_support_roles)
                and len({(role['native_name'], role['initialized_sha256'])
                         for role in fixed_support_roles}) == 1,
                'fixed support must keep one exact initialized physical actor')
    names = set().union(*map(set, active_sets))
    require(len(set(active_sets)) >= 2 and all(n.startswith('campaign_pool_') for n in names),
            'explicit distinct native pool identities required')
    common=None;initial={};states={};properties={};normalized=[]
    for ordinal,(document,active_values) in enumerate(zip(documents,active_sets)):
        active = set(active_values)
        init=document['objects_info']['init_info'];state=document['state']['registry']['object_registry']
        require(set(init)==set(state),'pool scene initialization/state inventory differs')
        pool=document.get('metadata',{}).get('ispatialgym',{}).get('model_pool',{})
        require(pool.get('manifest_sha256')==references[ordinal]['sha256']
                and active_actor_names(pool.get('active_names')) == active_values
                and set(pool.get('active_properties',{})) == active
                and {n for n in init if n.startswith('campaign_pool_')} == active,
                'source scene active/inactive pool inventory differs')
        fixed={n:v for n,v in init.items() if n not in active}
        if common is None:common=copy.deepcopy(fixed)
        require(common==fixed,'pool switch changes shared initialized geometry or robot')
        for name in active_values:
            if name in initial:
                require(initial[name] == init[name] and properties[name] == pool['active_properties'][name],
                        'same physical pool identity has different initialized properties')
            initial[name] = copy.deepcopy(init[name]); states.setdefault(name, copy.deepcopy(state[name]))
            properties[name] = copy.deepcopy(pool['active_properties'][name])
            args = init[name]['args']; flags = properties[name]
            require(args['category'] == flags['category'] and args['model'] == flags['model'] and
                    len(flags['scale']) == 3 and all(float(x)>0 for x in flags['scale'])
                    and flags['visible'] is True and bool(flags['links']),
                    'source active actor physical identity differs')
        doc=(normalize_active_role_bindings(document, active, role_maps[ordinal])
             if role_maps is not None else copy.deepcopy(document))
        if fixed_support_roles is not None:
            from .training_pool_source_compatibility import normalize_fixed_support
            doc = normalize_fixed_support(doc, active, fixed_support_roles[ordinal])
        doc['metadata']['ispatialgym'].pop('model_pool')
        # Every changed task binding must point to this explicitly selected actor.
        bindings=doc['metadata'].get('task',{}).get('inst_to_name',{})
        if role_maps is None:
            require(active <= set(bindings.values()), 'active actor lacks a native task binding')
            for key,value in list(bindings.items()):
                if value in active: bindings[key] = '<selected_pool_actor:'+key+'>'
        normalized.append(doc)
    checks=[compare_saved_scenes(normalized[0],doc,parking_objects=[],task_objects=names)
            for doc in normalized[1:]]
    union=copy.deepcopy(documents[0]);union['objects_info']['init_info'].update(initial)
    union['state']['registry']['object_registry'].update(states)
    result = {'schema':SCHEMA,'pool_manifest':pool_reference,'initialized_actors':initial,
            'source_actor_states':states,'active_properties':properties,
            'source_active_sequence':copy.deepcopy(list(active_names)),
            'static_scene_checks':checks,
            'union_scene':union,'production_released':False,'benchmark_credit':0,
            'scope':'Same scene/rooms and unchanged shared geometry; initialized model/scale actors only. '
                    'Room or support geometry changes require a separately verified contract.'}
    # Preserve the exact immutable v1 representation for old single-actor
    # source sequences. Explicit active sets use the additive field.
    if role_maps is not None or not all(isinstance(value, str) for value in active_names):
        result['source_active_sets'] = [list(values) for values in active_sets]
    if role_maps is not None:
        result['source_role_maps'] = copy.deepcopy(role_maps)
        result['original_task_metadata_sha256'] = [digest(doc['metadata']['task']) for doc in documents]
    if source_pool_references is not None:
        result['source_pool_manifests'] = copy.deepcopy(references)
    if fixed_support_roles is not None:
        result['source_fixed_support_roles'] = copy.deepcopy(fixed_support_roles)
    return result


def verify_selected_world(entry, record, document, manifest):
    """Compare campaign-qualified room IDs with native room-instance names."""
    expected = entry['expected']; scene = expected['scene']; room = expected['room']
    require(scene == manifest['scene'] and record['room'] == room,
            'selected world scene or recorded room differs')
    prefix = scene + '/'
    native_room = room[len(prefix):] if room.startswith(prefix) else room
    require('/' not in native_room and native_room in manifest['rooms'],
            'selected world room differs from exact native pool scope')
    args = document['init_info']['args']
    support = document['objects_info']['init_info'][expected['support']['native_name']]['args']
    require(args['scene_model'] == scene and native_room in args['load_room_instances']
            and native_room in support['in_rooms'],
            'selected native room or fixed support membership differs')


def prepare(root, request):
    """Validate the retained selected sequence and its exact physical sources."""
    from .selected_pool_episode_probe import inputs
    from ispatialgym.sampling.model_pool import read_manifest
    value,records=inputs(request)
    root=Path(root).resolve();require(root==Path(value['campaign_root']).resolve(),'campaign root differs')
    references=[r['template']['scene_setup']['native_model_pool'] for r in records]
    from .training_multi_active_sources import SCHEMA as MULTI_ACTIVE_SCHEMA, source_contract
    tasks = source_contract(value, records) if value['schema'] == MULTI_ACTIVE_SCHEMA else None
    pool_reference = references[0]; compatibility = None
    if not all(x == pool_reference for x in references):
        require(tasks is not None, 'sequence pool manifests differ')
        from .training_pool_source_compatibility import additive_manifest_union
        proof = checked(root, value['active_role_contract'])
        pool_reference, compatibility = additive_manifest_union(root, proof, references)
    manifest=checked(root,pool_reference)
    read_manifest(pool_reference['path'],scene=manifest['scene'],rooms=manifest['rooms'],
                  fixed_bindings=manifest['fixed_bindings'])
    require(all(r['task']==records[0]['task'] and
                (tasks is not None or (r['variant']==records[0]['variant'] and r['class_id']==records[0]['class_id']))
                and r['room']==records[0]['room']
                and r['perturbations']['lighting']==records[0]['perturbations']['lighting']
                and r['perturbations']['removed_objects']==records[0]['perturbations']['removed_objects']
                for r in records),'pool experiment changes undeclared task, room or augmentation')
    role_maps = [task['role_map'] for task in tasks] if tasks is not None else None
    names=([list(active_actor_names([v['native_name'] for v in roles.values()])) for roles in role_maps]
           if role_maps is not None else [e['expected']['target']['native_name'] for e in value['episodes']])
    docs=[checked(root,r['artifacts']['recorded_scene']) for r in records]
    support_roles = [task.get('fixed_support_role') for task in tasks] if tasks else None
    if support_roles is not None and not any(support_roles): support_roles = None
    contract=scene_contract(docs,names,pool_reference,role_maps=role_maps,
        source_pool_references=references if compatibility else None,
        fixed_support_roles=support_roles);members=[]
    if compatibility: contract['pool_manifest_compatibility'] = compatibility
    for ordinal,(entry,record,doc,active) in enumerate(zip(value['episodes'],records,docs,names)):
        tro=checked(root,record['artifacts']['tro'])['ispatialgym']
        roles = role_maps[ordinal] if role_maps is not None else {'target':entry['expected']['target']}
        for target in roles.values():
            binding=tro['object_bindings'][target['instance']]
            require(binding['scene_object_name']==target['native_name'] and binding['category']==target['category']
                    and binding['model']==target['model'],'selected TRO binding differs')
            flags=contract['active_properties'][target['native_name']]
            require(flags['scale']==target['scale'],'initialized actor scale differs from selected measured scale')
        verify_selected_world(entry, record, doc, manifest)
        members.append({'tro':record['artifacts']['tro'],'scene':record['artifacts']['recorded_scene'],
                        'active':active,'bindings':tro['object_bindings'],
                        'robot_poses':doc['metadata']['task']['robot_poses'],
                        'review':entry['review'],'configuration':entry['configuration']})
        if tasks is not None:
            members[-1]['task_definition'] = tasks[ordinal]
    contract.update(source_request={'path':str(Path(request).resolve()),'sha256':file_digest(request)},
                    native_runtime=value['native_runtime'],members=members,
                    selected_inputs_sha256=digest([r['validation_key'] for r in records]))
    return contract
