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


def isolated_member_scene(contract, document, active):
    """Build one explicit union initialization without changing a source scene.

    Inactive actors retain their exact initialized model/scale and serialized
    articulation state. Their roots start in separate out-of-room parking
    cells. Native collision/gravity/visibility isolation is a separate required
    operation before any episode restore; a parked JSON pose does not prove it.
    """
    require(contract.get('schema') == SCHEMA and active in contract['initialized_actors'],
            'unknown active pool member')
    names = set(contract['initialized_actors'])
    initial = document['objects_info']['init_info']
    states = document['state']['registry']['object_registry']
    require({n for n in initial if n.startswith('campaign_pool_')} == {active}
            and set(initial) == set(states)
            and initial[active] == contract['initialized_actors'][active],
            'member scene identity differs from checked pool contract')
    result = copy.deepcopy(document)
    result['objects_info']['init_info'].update(copy.deepcopy(contract['initialized_actors']))
    destinations = result['state']['registry']['object_registry']
    isolation = {}
    for ordinal, name in enumerate(sorted(names)):
        flags = copy.deepcopy(contract['active_properties'][name])
        if name != active:
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
        isolation[name] = {'active': name == active, 'properties': flags,
                           'state': copy.deepcopy(destinations[name])}
    require(set(destinations) == set(result['objects_info']['init_info']),
            'union scene object/state inventory differs')
    return result, isolation


def verify_pool_isolation(expected, observed, *, task_bindings, planner_names):
    """Require native properties and actual planner exclusion for every actor."""
    require(expected and expected.keys() == observed.keys(), 'native pool inventory differs')
    active = {name for name, row in expected.items() if row['active']}
    require(len(active) == 1 and active <= set(task_bindings.values()),
            'selected actor lacks its exact task binding')
    inactive = expected.keys() - active
    require(not inactive.intersection(task_bindings.values()), 'inactive actor is task-bound')
    for name, row in expected.items():
        actual = observed[name]
        require(actual['properties'] == row['properties'],
                'native pool physical properties differ: '+name)
        if not row['active']:
            require(name not in planner_names and not actual.get('assisted_held')
                    and not actual.get('attached_parent')
                    and math.dist(actual['position'], row['state']['root_link']['pos']) <= 1e-6
                    and all(abs(float(x)) <= 1e-7 for k in ('linear_velocity', 'angular_velocity')
                            for x in actual[k]),
                    'inactive actor remains physically or logically active: '+name)
    return {'active': sorted(active), 'inactive': sorted(inactive),
            'properties_verified': True, 'planner_exclusion_verified': True,
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


def scene_contract(documents, active_names, pool_reference):
    """Verify shared static geometry, allowing only explicitly selected actors.

    Source scenes remain immutable. The returned union is a new, declared
    experimental initialization artifact, never substituted for source exports.
    """
    require(len(documents)==len(active_names)>=2,'pool contract needs distinct member scenes')
    names=set(active_names)
    require(len(names)>=2 and all(n.startswith('campaign_pool_') for n in names),
            'explicit distinct native pool identities required')
    common=None;initial={};states={};properties={};normalized=[]
    for document,active in zip(documents,active_names):
        init=document['objects_info']['init_info'];state=document['state']['registry']['object_registry']
        require(set(init)==set(state),'pool scene initialization/state inventory differs')
        pool=document.get('metadata',{}).get('ispatialgym',{}).get('model_pool',{})
        require(pool.get('manifest_sha256')==pool_reference['sha256']
                and pool.get('active_names')==[active]
                and set(pool.get('active_properties',{}))=={active}
                and {n for n in init if n.startswith('campaign_pool_')}=={active},
                'source scene active/inactive pool inventory differs')
        fixed={n:v for n,v in init.items() if n!=active}
        if common is None:common=copy.deepcopy(fixed)
        require(common==fixed,'pool switch changes shared initialized geometry or robot')
        if active in initial:
            require(initial[active]==init[active] and properties[active]==pool['active_properties'][active],
                    'same physical pool identity has different initialized properties')
        initial[active]=copy.deepcopy(init[active]);states.setdefault(active,copy.deepcopy(state[active]))
        properties[active]=copy.deepcopy(pool['active_properties'][active])
        args=init[active]['args'];flags=properties[active]
        require(args['category']==flags['category'] and args['model']==flags['model'] and
                len(flags['scale'])==3 and all(float(x)>0 for x in flags['scale'])
                and flags['visible'] is True and bool(flags['links']),
                'source active actor physical identity differs')
        doc=copy.deepcopy(document)
        doc['metadata']['ispatialgym'].pop('model_pool')
        # Every changed task binding must point to this explicitly selected actor.
        bindings=doc['metadata'].get('task',{}).get('inst_to_name',{})
        require(active in bindings.values(),'active actor lacks a native task binding')
        for key,value in list(bindings.items()):
            if value==active:bindings[key]='<selected_pool_actor>'
        normalized.append(doc)
    checks=[compare_saved_scenes(normalized[0],doc,parking_objects=[],task_objects=names)
            for doc in normalized[1:]]
    union=copy.deepcopy(documents[0]);union['objects_info']['init_info'].update(initial)
    union['state']['registry']['object_registry'].update(states)
    return {'schema':SCHEMA,'pool_manifest':pool_reference,'initialized_actors':initial,
            'source_actor_states':states,'active_properties':properties,
            'source_active_sequence':list(active_names),'static_scene_checks':checks,
            'union_scene':union,'production_released':False,'benchmark_credit':0,
            'scope':'Same scene/rooms and unchanged shared geometry; initialized model/scale actors only. '
                    'Room or support geometry changes require a separately verified contract.'}


def prepare(root, request):
    """Validate the retained selected sequence and its exact physical sources."""
    from .selected_pool_episode_probe import inputs
    from ispatialgym.sampling.model_pool import read_manifest
    value,records=inputs(request)
    root=Path(root).resolve();require(root==Path(value['campaign_root']).resolve(),'campaign root differs')
    references=[r['template']['scene_setup']['native_model_pool'] for r in records]
    require(all(x==references[0] for x in references),'sequence pool manifests differ')
    manifest=checked(root,references[0])
    read_manifest(references[0]['path'],scene=manifest['scene'],rooms=manifest['rooms'],
                  fixed_bindings=manifest['fixed_bindings'])
    require(all(r['task']==records[0]['task'] and r['variant']==records[0]['variant']
                and r['room']==records[0]['room'] and r['class_id']==records[0]['class_id']
                and r['perturbations']['lighting']==records[0]['perturbations']['lighting']
                and r['perturbations']['removed_objects']==records[0]['perturbations']['removed_objects']
                for r in records),'pool experiment changes undeclared task, room or augmentation')
    names=[e['expected']['target']['native_name'] for e in value['episodes']]
    docs=[checked(root,r['artifacts']['recorded_scene']) for r in records]
    contract=scene_contract(docs,names,references[0]);members=[]
    for entry,record,doc,active in zip(value['episodes'],records,docs,names):
        tro=checked(root,record['artifacts']['tro'])['ispatialgym']
        target=entry['expected']['target'];binding=tro['object_bindings'][target['instance']]
        require(binding['scene_object_name']==active and binding['category']==target['category']
                and binding['model']==target['model'],'selected TRO binding differs')
        require(entry['expected']['scene']==manifest['scene'] and
                entry['expected']['room'] in manifest['rooms'],'selected world scope differs')
        flags=contract['active_properties'][active]
        require(flags['scale']==target['scale'],'initialized actor scale differs from selected measured scale')
        members.append({'tro':record['artifacts']['tro'],'scene':record['artifacts']['recorded_scene'],
                        'active':active,'bindings':tro['object_bindings'],
                        'robot_poses':doc['metadata']['task']['robot_poses'],
                        'review':entry['review'],'configuration':entry['configuration']})
    contract.update(source_request={'path':str(Path(request).resolve()),'sha256':file_digest(request)},
                    native_runtime=value['native_runtime'],members=members,
                    selected_inputs_sha256=digest([r['validation_key'] for r in records]))
    return contract
