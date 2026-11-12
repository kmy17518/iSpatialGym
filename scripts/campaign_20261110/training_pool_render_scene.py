"""Checked fixed-inventory scene documents for a bounded native pool renderer.

These are private renderer inputs. Source recordings remain active-only; no
function here installs a renderer hook or authorizes a native job.
"""
import copy
import math

from .contracts import require, digest

SCHEMA = 'isg-pool-render-scene-union-v1'


def _robot_geometry(info):
    result = copy.deepcopy(info)
    args = result['args']
    require(('position' in args) == ('orientation' in args), 'renderer robot initialization pose is partial')
    if 'position' not in args: return result
    for key, size in (('position', 3), ('orientation', 4)):
        value = args.pop(key, None)
        require(isinstance(value, list) and len(value) == size
                and all(type(v) in (int, float) and math.isfinite(v) for v in value),
                'renderer robot initialization pose missing or invalid')
    return result


def scene_union(documents, active_names, robot_name, physical_contract):
    require(len(documents) == len(active_names) == 2 and len(set(active_names)) == 2
            and set(active_names) == set(physical_contract['initialized_actors']),
            'renderer needs the exact two physically verified pool actors')
    actors = set(active_names); shared = None; pool_init = {}; source_states = {}; scene_geometry = None
    for doc, active in zip(documents, active_names):
        init = doc['objects_info']['init_info']; states = doc['state']['registry']['object_registry']
        require(set(init) == set(states) and set(init).intersection(actors) == {active}
                and robot_name in init, 'source renderer scene is not active-only and complete')
        meta = doc['metadata']['ispatialgym']['model_pool']
        require(meta['active_names'] == [active]
                and meta['active_properties'] == {active: physical_contract['active_properties'][active]},
                'renderer actor model/scale/static properties differ from physical proof')
        background = {name: copy.deepcopy(info) for name, info in init.items() if name not in actors}
        background[robot_name] = _robot_geometry(background[robot_name])
        if shared is None: shared = background
        require(background == shared, 'renderer shared initialized geometry or controller configuration differs')
        geometry = copy.deepcopy(doc['init_info']); geometry['args'].pop('scene_file')
        if scene_geometry is None: scene_geometry = geometry
        require(geometry == scene_geometry, 'renderer scene, room load, or support boundary differs')
        pool_init[active] = copy.deepcopy(init[active]); source_states[active] = copy.deepcopy(states[active])
    canonical = {name: copy.deepcopy(info) for name, info in documents[0]['objects_info']['init_info'].items()
                 if name not in actors}
    canonical.update({name: pool_init[name] for name in sorted(actors)})
    return {'schema': SCHEMA, 'initialized_actors': canonical, 'pool_names': sorted(actors),
            'source_actor_states': source_states, 'robot_name': robot_name,
            'physical_properties': copy.deepcopy(physical_contract['active_properties']),
            'source_scene_digests': [digest(doc) for doc in documents],
            'shared_geometry_sha256': digest(shared), 'scene_geometry_sha256': digest(scene_geometry),
            'scope': 'Exact recorded same-room/support geometry only; no new room/support transition proof.',
            'renderer_native_verified': False, 'production_released': False}


def member_scene(contract, document, active):
    require(contract['schema'] == SCHEMA and active in contract['pool_names']
            and digest(document) in contract['source_scene_digests'], 'unbound renderer scene source')
    result = copy.deepcopy(document); names = set(contract['pool_names'])
    require(set(result['objects_info']['init_info']).intersection(names) == {active},
            'renderer member identity differs')
    result['objects_info']['init_info'] = copy.deepcopy(contract['initialized_actors'])
    states = result['state']['registry']['object_registry']; parking = {}
    for index, name in enumerate(contract['pool_names']):
        if name == active: continue
        state = copy.deepcopy(contract['source_actor_states'][name]); root = state['root_link']
        require(set(root) >= {'pos', 'ori', 'lin_vel', 'ang_vel'}, 'inactive recorded actor lacks native root state')
        root.update(pos=[40.+index, 40., 5.], lin_vel=[0.,0.,0.], ang_vel=[0.,0.,0.])
        states[name] = state; parking[name] = list(root['pos'])
    require(set(states) == set(contract['initialized_actors']), 'renderer union inventory differs')
    result['state']['registry']['object_registry'] = {name: states[name] for name in contract['initialized_actors']}
    return result, parking


def observe_visibility(scene, contract, active, parking):
    """Read native inclusion; the caller must have applied the stated visibility."""
    observed = {}
    for name in contract['pool_names']:
        obj = scene.object_registry('name', name)
        require(obj is not None and bool(obj.visible) == (name == active), 'renderer active visibility differs')
        expected = contract['physical_properties'][name]
        require(obj.category == expected['category'] and obj.model == expected['model']
                and [float(v) for v in obj.scale] == expected['scale'],
                'renderer native actor model/category/scale differs')
        position = [float(v) for v in obj.get_position_orientation()[0]]
        if name != active:
            require(len(position) == 3 and math.dist(position, parking[name]) <= 1e-5,
                    'renderer inactive actor left its explicit parking cell')
        observed[name] = {'active': name == active, 'visible': bool(obj.visible), 'position': position,
                          'category': obj.category, 'model': obj.model, 'scale': [float(v) for v in obj.scale]}
    return observed
