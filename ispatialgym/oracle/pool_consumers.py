"""Per-episode planner and recording consumers for an already loaded actor pool.

This module does not load actors, mutate physics, or grant execution authority.
The native loader must bind an exact checked member and initialize the declared
union before these consumers can run. Internal checkpoints retain every actor.
"""
import copy
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

from ispatialgym.sampling.model_pool import require, file_reference


def active_actor_names(value):
    values = [value] if isinstance(value, str) else value
    require(isinstance(values, (list, tuple)) and values and
            all(isinstance(v, str) and v for v in values) and len(set(values)) == len(values),
            'active actor inventory is empty or ambiguous')
    return tuple(values)

def excluding_inactive(previous, initialized, active):
    initialized, active = set(initialized), set(active)
    require(active and active <= initialized, 'consumer active inventory differs')
    require(previous is None or callable(previous), 'consumer filter is invalid')
    inactive = initialized-active
    return lambda obj: obj.name not in inactive and (previous is None or bool(previous(obj)))


def _list(value):
    if hasattr(value, 'detach'): value = value.detach().cpu()
    return value.tolist() if hasattr(value, 'tolist') else value


def planner_observation(backend, objects, active):
    """Read native masks immediately after full sync and before grasp attachment.

    Cached inactive meshes may remain allocated but must be disabled. Every
    active actor collider must be present and enabled at this specific boundary.
    This is not a planner path or collision-clearance success claim.
    """
    active = set(active_actor_names(active))
    require(objects and active <= set(objects) and all(k == v.name for k,v in objects.items()),
            'planner pool inventory differs')
    store = backend.checker.data.meshes
    names, enabled, counts = store.names, _list(store.enable), _list(store.count)
    require(len(names) == len(enabled) == len(counts) == 1 and len(names[0]) == len(enabled[0]),
            'one native planner environment and complete mask required')
    count = counts[0]
    require(type(count) is int and 0 <= count <= len(names[0])
            and all(x in (0,1) for x in enabled[0])
            and all(x == 0 for x in enabled[0][count:]), 'invalid native mesh allocation/mask')
    slots = {}
    for index, name in enumerate(names[0][:count]):
        require(isinstance(name,str) and name and name not in slots, 'ambiguous native mesh name')
        slots[name] = bool(enabled[0][index])
    observations = {}
    for name, obj in objects.items():
        paths = [str(mesh.prim_path) for link in obj.links.values() for mesh in link.collision_meshes.values()]
        require(paths and len(paths) == len(set(paths)), 'pool collider inventory absent or duplicated')
        rows = {path:{'allocated':path in slots,'enabled':slots.get(path,False)} for path in paths}
        require(all(r['allocated'] and r['enabled'] for r in rows.values()) if name in active
                else not any(r['enabled'] for r in rows.values()),
                'native pool planner participation differs: '+name)
        observations[name] = {'active':name in active,'colliders':rows}
    return {'boundary':'after_full_world_sync_before_grasp_attachment','objects':observations,
            'native_enable_mask_read':True,'allocated_mesh_count':count,
            'enabled_mesh_count':sum(enabled[0]),'trajectory_collision_clearance_verified':False}


@contextmanager
def recorded_registry_scope(scene, initialized, active, *, initial_state):
    """Restrict one explicit recording call, preserving the exact prior filter.

    Initial recorded state includes all non-pool actors even if asleep or
    unchanged. Later recorded states retain the original changed/awake filter.
    Internal loader/checkpoint snapshots must not enter this recording scope.
    """
    registry = scene.object_registry
    require(hasattr(registry,'_dump_filter') and callable(getattr(registry,'set_dump_filter',None)),
            'native recording registry filter unavailable')
    prior = registry._dump_filter
    callback = excluding_inactive(None if initial_state else prior, initialized, active)
    registry.set_dump_filter(callback)
    try:
        yield
    finally:
        registry.set_dump_filter(prior)


def serialized_object_names(state):
    """Accept native scene or simulator layout, with exactly one scene registry."""
    registries = []
    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == 'object_registry':
                    require(isinstance(child, dict), 'native object registry invalid')
                    registries.append(child)
                else: visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value: visit(child)
    visit(state)
    require(len(registries) == 1, 'recorded state must contain exactly one native scene')
    return set(registries[0])


def verify_serialized(owner, flat, initialized, active, all_scene_names, initial):
    state, consumed = owner.deserialize(flat)
    require(int(consumed) == int(flat.numel()), 'native serialized state has unread trailing data')
    names = serialized_object_names(state); inactive = set(initialized)-set(active_actor_names(active))
    require(not names.intersection(inactive) and names <= set(all_scene_names),
            'recorded state contains inactive or unknown actors')
    if initial:
        require(names == set(all_scene_names)-inactive, 'initial state omitted background or active actors')
    return {'initial': initial, 'native_reader': type(owner).__name__, 'elements': int(flat.numel()),
            'object_names': sorted(names), 'inactive_names_absent': True,
            'serialized_float32_sha256': hashlib.sha256(flat.detach().cpu().float().contiguous().numpy().tobytes()).hexdigest()}


def active_document(document, source, source_metadata, initialized, active, properties, bindings, robot_poses,
                    initial_scene_path, exported_scene_path):
    from ispatialgym.sampling.model_pool import active_scene
    result = copy.deepcopy(document)
    # The initialized pool's native snapshot inherits the complete known
    # source metadata under task, then overlays current physical bindings and
    # robot poses. Preserve its full lineage and reject every unknown field.
    require(set(result['metadata']) == {'task'}, 'unknown native metadata wrapper')
    native = result['metadata']['task']
    inherited = {k: v for k, v in native.items() if k not in ('inst_to_name', 'robot_poses')}
    require(any(inherited == value for value in source_metadata)
            and native['inst_to_name'] == bindings and native['robot_poses'] == robot_poses,
            'native metadata is not a known source plus current task fields')
    result['metadata'] = copy.deepcopy(source['metadata'])
    result['metadata']['task'].update(inst_to_name=copy.deepcopy(bindings), robot_poses=copy.deepcopy(robot_poses))
    target = result['metadata']['ispatialgym']['model_pool']
    expected = source['metadata']['ispatialgym']['model_pool']
    names = active_actor_names(active)
    measured = {active: properties} if isinstance(active, str) else properties
    require(set(target) == set(expected) == {'manifest_sha256', 'batch_id', 'active_names',
                                           'inactive_omitted', 'active_properties'}
            and all(target[k] == expected[k] for k in ('manifest_sha256', 'batch_id'))
            and active_actor_names(expected['active_names']) == names
            and set(expected['active_properties']) == set(names)
            and measured == expected['active_properties']
            and set(names) <= set(bindings.values())
            and result['metadata']['task']['inst_to_name'] == bindings,
            'native scene metadata or current task binding differs')
    target.update(active_names=list(names), inactive_omitted=len(initialized)-len(names),
                  active_properties=copy.deepcopy(measured))
    require(result['init_info']['args']['scene_file'] == str(initial_scene_path),
            'native initialization scene differs from recorded environment')
    result['init_info']['args']['scene_file'] = str(exported_scene_path)
    return active_scene(result, initialized, set(names))


def recorded_config(config, scene_path, tro_path, nominal_image_size, native_defaults=None):
    def merge(base, update):
        result = copy.deepcopy(base)
        for key, value in update.items():
            result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
        return result
    result = merge(native_defaults or {}, config)
    require(result['task']['tro_path'] == tro_path and len(result['robots']) == 1,
            'recording config does not name current member')
    result['scene']['scene_file'] = str(scene_path)
    if 'num_envs' in result['env']: result['env']['num_envs'] = 1
    if nominal_image_size:
        for robot in result['robots']:
            kw = ((robot.get('sensor_config') or {}).get('VisionSensor') or {}).get('sensor_kwargs')
            if isinstance(kw, dict):
                for name in ('image_height', 'image_width'):
                    if name in kw: kw[name] = int(nominal_image_size)
    return result




class PoolConsumers:
    """One source's exact active set; never reused as mutable cross-episode state."""

    def __init__(self, contract, tro_path):
        self.contract = copy.deepcopy(contract)
        self.initialized = frozenset(contract['initialized_actors'])
        require(self.initialized and all(isinstance(n, str) and n for n in self.initialized),
                'initialized pool inventory is invalid')
        tro_path = str(Path(tro_path).resolve())
        members = [m for m in contract['members'] if m['tro']['path'] == tro_path]
        require(members and all(m == members[0] for m in members),
                'exact incoming pool member is absent or ambiguous')
        self.member = copy.deepcopy(members[0])
        self.active = active_actor_names(self.member['active'])
        require(set(self.active) <= self.initialized, 'incoming active actors are not initialized')
        self.known_metadata = []
        for member in contract['members']:
            for reference in (member['tro'], member['scene']):
                require(file_reference(reference['path']) == reference, 'pool source changed')
            self.known_metadata.append(json.loads(Path(member['scene']['path']).read_text())['metadata'])
        self.source = json.loads(Path(self.member['scene']['path']).read_text())
        self.recording_config = None
        self.loader = None
        self.observations = []
        self.planner = None

    def bind_loader(self, loader):
        require(self.loader is None and str(Path(loader.tro_path).resolve()) == self.member['tro']['path'],
                'pool consumers belong to one exact incoming loader')
        self.loader = loader

    def ready(self, *, verify_sources=False):
        require(self.loader is not None and self.loader.env is not None,
                'pool consumer has no loaded world')
        if verify_sources:
            require(file_reference(self.member['tro']['path']) == self.member['tro'] and
                    file_reference(self.member['scene']['path']) == self.member['scene'],
                    'incoming source changed after binding')
        scene = self.loader.env.scene
        require(self.initialized <= set(scene.object_registry.get_dict('name')),
                'loaded world omits initialized pool actors')
        return scene

    def obstacle_filter(self, previous):
        self.ready()
        return excluding_inactive(previous, self.initialized, self.active)

    def observe_planner(self, backend):
        scene = self.ready()
        require(backend.robot is self.loader.robot and not backend._attached,
                'pool planner must belong to this world before grasp attachment')
        objects = {name: scene.object_registry('name', name) for name in self.initialized}
        self.planner = planner_observation(backend, objects, self.active)
        self.loader.report['model_pool_planner'] = copy.deepcopy(self.planner)
        return self.planner

    def dump_recording(self, executor, *, initial):
        scene = self.ready()
        require(executor.env is self.loader.env, 'recording belongs to a different pool world')
        with recorded_registry_scope(scene, self.initialized, self.active, initial_state=initial):
            flat = executor.state_owner.dump_state(serialized=True)
        evidence = verify_serialized(executor.state_owner, flat, self.initialized, self.active,
                                     scene.object_registry.get_dict('name'), initial)
        self.observations.append(evidence)
        return flat

    def prepare_recording(self, config):
        self.ready(verify_sources=True)
        require(config['task']['tro_path'] == self.member['tro']['path'] and
                len(config['robots']) == 1, 'recording configuration is not the incoming member')
        self.recording_config = copy.deepcopy(config)

    def snapshot(self, recorder):
        from ispatialgym.sampling.model_pool import native_properties
        scene = self.ready(verify_sources=True)
        require(recorder.env is self.loader.env and self.recording_config is not None,
                'pool snapshot has no exact incoming configuration')
        require(not self.loader.external_camera and not getattr(self.loader, 'behavior_output', False),
                'external/behavior pool recording requires its own reviewed contract')
        document = recorder.scene_file
        if isinstance(document, str):
            document = json.loads(document) if document.lstrip().startswith('{') else json.loads(Path(document).read_text())
        require(isinstance(document, dict) and isinstance(recorder.env_config, dict),
                'native pool scene snapshot failed')
        bindings = {k: v.name for k, v in self.loader.task.physical_object_scope.items() if v is not None}
        properties = {name: native_properties(scene.object_registry('name', name)) for name in self.active}
        if isinstance(self.member['active'], str): properties = properties[self.member['active']]
        initial = Path(self.loader.cfg['scene']['scene_file']).resolve()
        require(Path(recorder.env_config['scene']['scene_file']).resolve() == initial,
                'native recording names a different initialized world')
        destination = Path(recorder.out_dir).resolve()/'recorded_active_scene.json'
        require(not destination.exists(), 'active pool recording output already exists')
        projected = active_document(document, self.source, self.known_metadata,
            self.initialized, self.member['active'], properties, bindings,
            self.loader.tro.robot_poses, initial, destination)
        config = recorded_config(self.recording_config, destination, self.member['tro']['path'],
                                 recorder.nominal_image_size, recorder.env_config)
        require(not any(name in json.dumps(config) for name in self.initialized-set(self.active)),
                'inactive actor referenced by recorded configuration')
        # Create exclusively: retrying an output path must not overwrite evidence.
        with destination.open('x') as stream:
            json.dump(projected, stream, indent=2)
            stream.write('\n')
        recorder.scene_file, recorder.env_config = projected, config
        self.loader.report['model_pool_recording'] = {
            'active_scene': file_reference(destination), 'source_tro': self.member['tro'],
            'active_names': list(self.active), 'initialized_names': sorted(self.initialized),
            'internal_snapshots': 'full_initialized_union', 'physical_scope_bindings': bindings}
