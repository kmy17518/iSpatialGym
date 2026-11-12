"""Explicit one-world batches whose only scene variation is light and parking.

This contract does not change the ordinary template-compatibility rule. It is
restricted to fixed-base, non-task actors already present in both native scenes.
Every member still needs its own immutable native/demo/replay evidence.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from scripts.campaign_20261110.contracts import digest, file_digest, require
from ispatialgym.sampling.campaign_augmentation import checked_removals, validate_lighting

SCHEMA = 'isg-sequential-augmentation-contract-v1'


def _read(path):
    return json.loads(Path(path).read_text())


def _template_identity(record):
    template = copy.deepcopy(record['template'])
    template.pop('template_id', None)
    setup = template['scene_setup']
    # These identify the declared augmentation draw, not a different asset/task.
    setup.pop('augmentation_probe', None)
    setup.pop('training_scale_plan_sha256', None)
    return template


def record_scope(records):
    """CPU-only compatibility audit, before importing or modifying a simulator."""
    require(len(records) == 2 and records[0]['instance_id'] != records[1]['instance_id'],
            'sequential augmentation needs two distinct validated inputs')
    a, b = records
    require(_template_identity(a) == _template_identity(b), 'sequential batch physical template differs')
    for field in ('task', 'class_id', 'room', 'variant', 'sampling_split'):
        require(a[field] == b[field], 'sequential batch task/room differs: ' + field)
    keys = [{k: v for k, v in r['validation_key'].items() if k != 'template_sha256'} for r in records]
    require(keys[0] == keys[1], 'sequential batch runtime/data/physics identity differs')
    perturbations = [{k: v for k, v in r['perturbations'].items()
                      if k not in ('lighting', 'removed_objects')} for r in records]
    require(perturbations[0] == perturbations[1], 'sequential batch non-augmentation perturbation differs')
    def stable_reference(row):
        setup=row['template']['scene_setup']
        return setup.get('sequential_augmentation_base', {}).get('scene',
            setup['training_height_dependency']['stable_scene'])
    stable_ref = stable_reference(a)
    require(stable_ref == stable_reference(b) and
            file_digest(stable_ref['path']) == stable_ref['sha256'], 'sequential batch stable scene changed')
    stable = _read(stable_ref['path'])
    protected = set(a['template']['scene_setup']['inst_to_name'].values())
    parked = set()
    for row in records:
        validate_lighting(row['perturbations']['lighting'])
        names = row['perturbations']['removed_objects']
        require(names == checked_removals(stable, names, protected), 'sequential parking names differ')
        parked.update(names)
    require(a['perturbations']['lighting'] != b['perturbations']['lighting'] and
            a['perturbations']['removed_objects'] != b['perturbations']['removed_objects'],
            'sequential augmentation coverage requires changed lighting and parking')
    require(parked and all(stable['objects_info']['init_info'][n]['args'].get('fixed_base') is True
                          for n in parked), 'sequential parking requires originally fixed-base actors')
    return {'schema': SCHEMA, 'stable_scene': stable_ref, 'parking_objects': sorted(parked),
            'protected_objects': sorted(protected), 'slots': 1,
            'members': [{'instance_id': r['instance_id'], 'tro': r['artifacts']['tro'],
                         'perturbations_sha256': digest(r['perturbations'])} for r in records]}


def _leaves(value, prefix=''):
    if isinstance(value, dict):
        return {p: v for k, x in value.items() for p, v in _leaves(x, prefix+'/'+str(k)).items()}
    if isinstance(value, list):
        return {p: v for i, x in enumerate(value) for p, v in _leaves(x, prefix+'['+str(i)+']').items()}
    return {prefix: value}


def compare_saved_scenes(first, second, *, parking_objects, task_objects, tolerance=.005):
    """Retain the native static-scene tolerance; omit only declared state writes."""
    normalized = []
    for source in (first, second):
        doc = copy.deepcopy(source)
        init = doc['objects_info']['init_info']; states = doc['state']['registry']['object_registry']
        require(set(init) == set(states), 'saved scene initialization/state inventory differs')
        for name in parking_objects:
            require(name in init and init[name]['args'].get('fixed_base') is True,
                    'sequential parking actor missing or not fixed')
        for name in list(init):
            if name in task_objects or name.startswith('robot'):
                init.pop(name); states.pop(name)
        # Only poses/velocities of the declared fixed parking actors may differ.
        for name in parking_objects:
            root = states[name]['root_link']
            for field in ('pos', 'ori', 'lin_vel', 'ang_vel'):
                root.pop(field, None)
        meta = doc.get('metadata', {})
        meta.get('task', {}).pop('robot_poses', None)
        for key in ('moved_objects', 'removed_objects', 'source_scene_file'):
            meta.get('ispatialgym', {}).pop(key, None)
        doc.get('init_info', {}).get('args', {}).pop('scene_file', None)
        normalized.append({p: v for p, v in _leaves(doc).items()
                           if not (p.endswith('/is_asleep') or '/lin_vel[' in p or
                                   '/ang_vel[' in p or '/non_kin/AABB/' in p)})
    a, b = normalized
    require(set(a) == set(b), 'sequential saved scene entry inventory differs')
    worst = 0.
    for path, x in a.items():
        y = b[path]
        if type(x) in (int, float) and type(y) in (int, float):
            require(math.isfinite(x) and math.isfinite(y), 'nonfinite saved scene value')
            worst = max(worst, abs(x-y))
            require(abs(x-y) <= tolerance, 'undeclared static scene change: ' + path)
        else:
            require(x == y, 'undeclared saved scene identity change: ' + path)
    return {'static_entries_checked': len(a), 'maximum_numeric_difference': worst,
            'native_static_tolerance': tolerance}


def checked_nested_parking_provenance(scene, template_meta, stable_scene):
    """Remove only native-bound parking history, never loaded actors or state.

    Native exports nest scene provenance under metadata.task.ispatialgym. A
    parked variant records its own operation; an unmodified variant can retain
    earlier stable-scene history for actors outside the loaded crop.
    """
    document = copy.deepcopy(scene)
    actual = document.get('metadata', {}).get('task', {}).get('ispatialgym', {})
    fields = ('moved_objects', 'removed_objects')
    if not any(key in actual for key in fields):
        return document
    variant = template_meta.get('scene_variant')
    baseline = stable_scene.get('metadata', {})
    candidates = [variant] if variant else [baseline.get('ispatialgym', {}),
        baseline.get('task', {}).get('ispatialgym', {})]
    observed = {key: actual[key] for key in fields if key in actual}
    require(any(isinstance(candidate, dict) and
                observed == {key: candidate[key] for key in fields if key in candidate}
                for candidate in candidates), 'saved nested parking provenance differs from its native source')
    require(isinstance(actual.get('moved_objects', {}), dict) and
            isinstance(actual.get('removed_objects', []), list), 'invalid nested parking provenance')
    for key in fields:
        actual.pop(key, None)
    return document


def saved_world_scope(runner, records):
    contract = record_scope(records)
    paths = [r['artifacts']['tro']['path'] for r in records]
    resolved = [runner.resolve_instance(path) for path in paths]
    signatures = [json.loads(runner.instance_signature(path)) for path in paths]
    require([{k: v for k, v in s.items() if k != 'template'} for s in signatures][0] ==
            [{k: v for k, v in s.items() if k != 'template'} for s in signatures][1],
            'sequential constraints or attachments differ')
    metas = [x[2] for x in resolved]
    require(all(metas[0].get(k) == metas[1].get(k) for k in
                ('scene', 'load_room_instances', 'problem', 'scene_crop')),
            'sequential native task/crop/room differs')
    definitions = [Path(x[1])/'activity_definitions'/x[3]/('problem'+str(x[4])+'.bddl') for x in resolved]
    require(file_digest(definitions[0]) == file_digest(definitions[1]), 'sequential native BDDL differs')
    raw = [_read(path)['ispatialgym'] for path in paths]
    require(raw[0]['object_bindings'] == raw[1]['object_bindings'], 'sequential native task bindings differ')
    task_objects = set(runner._tro_object_names(paths[0]))
    require(task_objects == set(runner._tro_object_names(paths[1])) and
            not task_objects.intersection(contract['parking_objects']), 'sequential parking overlaps task objects')
    scenes = [Path(runner.loaded_scene_path(x[1], x[2])) for x in resolved]
    docs = [_read(path) for path in scenes]
    stable_scene = _read(contract['stable_scene']['path'])
    stable_states = stable_scene['state']['registry']['object_registry']
    compared = [checked_nested_parking_provenance(doc, meta, stable_scene) for doc, meta in zip(docs, metas)]
    comparison = compare_saved_scenes(*compared, parking_objects=contract['parking_objects'], task_objects=task_objects)
    members = {}
    for path, scene, doc, meta, row in zip(paths, scenes, docs, metas, records):
        moved = meta.get('scene_variant', {}).get('moved_objects', {})
        names = row['perturbations']['removed_objects']
        # Actual native scene bytes, not an invented transform, supply every pose.
        states = doc['state']['registry']['object_registry']
        for name in contract['parking_objects']:
            native = states[name]['root_link']; baseline = stable_states[name]['root_link']
            if name in names:
                require(name in moved and moved[name]['fixed_base'] is True and
                        math.dist(moved[name]['pos'], baseline['pos']) <= .002,
                        'native parking origin differs from declared stable scene')
                expected = moved[name]['moved_to']
            else:
                expected = baseline['pos']
            require(math.dist(native['pos'], expected) <= .002,
                    'saved parking/restored pose differs from native augmentation metadata')
            qa, qb = native['ori'], baseline['ori']
            norms = math.sqrt(sum(float(x)**2 for x in qa)*sum(float(x)**2 for x in qb))
            require(norms > 0 and 2*math.acos(min(1.,abs(sum(float(x)*float(y)for x,y in zip(qa,qb)))/norms)) <= 1e-4,
                    'sequential parking changed actor orientation')
        members[str(Path(path).resolve())] = {'tro': row['artifacts']['tro'],
            'scene': {'path': str(scene), 'sha256': file_digest(scene)},
            'parking_states': {name: copy.deepcopy(states[name]) for name in contract['parking_objects']}}
    return {**contract, 'members_by_tro': members, 'saved_scene_check': comparison,
            'compatible_template_dirs': [x[1] for x in resolved],
            'method': 'exact task/assets/runtime plus declared fixed-actor parking and lighting'}


def restore_parking(loader, contract):
    """Restore declared native actor states; add no simulation or settling step."""
    require(contract.get('schema') == SCHEMA, 'invalid sequential augmentation scope')
    member = contract['members_by_tro'].get(str(Path(loader.tro_path).resolve()))
    require(member is not None and file_digest(loader.tro_path) == member['tro']['sha256'] and
            file_digest(member['scene']['path']) == member['scene']['sha256'],
            'sequential episode input changed')
    restored = []
    for name, state in member['parking_states'].items():
        obj = loader.env.scene.object_registry('name', name)
        require(obj is not None and obj.fixed_base, 'sequential parking actor runtime identity differs')
        obj.load_state(copy.deepcopy(state), serialized=False)
        restored.append(name)
    return {'schema': SCHEMA, 'source_scene': member['scene'], 'objects': restored,
            'physics_steps_added': 0}
