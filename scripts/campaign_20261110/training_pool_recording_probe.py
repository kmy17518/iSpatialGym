"""Active-only diagnostic recordings without filtering internal pool snapshots."""
import copy
import hashlib
import json
from pathlib import Path

from .contracts import require, file_digest, write_json_atomic

SCHEMA = 'isg-pool-active-recording-runtime-v1'
PARENT_COMMIT = '53b0cc9274b94e29b44ce2cadbc9269a2f7d1339'
DELTA = {'name': 'pool_active_recording_v1', 'source_inputs_changed': False,
         'load_restore_operations_changed': False, 'native_actions_changed': False,
         'thresholds_changed': False, 'extra_physics_steps': 0,
         'initial_recording': 'all_background_and_current_pool_actor',
         'step_recording': 'ordinary_filter_composed_with_inactive_exclusion',
         'internal_full_snapshots': 'unchanged_full_initialized_union',
         'scene_export': 'native_snapshot_projected_to_current_active_actor',
         'configuration': 'native_current_member_builder_and_nominal_camera_size',
         'serialized_reader': 'native_owner_deserialize_every_recorded_state',
         'renderer_equivalence_verified': False}
ALLOWED = {'scripts/campaign_20261110/training_pool_recording_probe.py',
           'scripts/campaign_20261110/tests/test_training_pool_recording_probe.py',
           'scripts/campaign_20261110/training_pool_replay_probe.py',
           'scripts/campaign_20261110/training_pool_probe.py',
           'scripts/campaign_20261110/training_pool_recovery_probe.py',
           'scripts/campaign_20261110/tests/test_training_pool_recovery_probe.py'}


def parent_request(request, proof=None):
    require(request.get('pool_active_recording_v1') is True and request.get('pool_action_replay_v1') is True
            and request.get('diagnostic_control') in ('FRESH_A', 'FRESH_B', 'ABA')
            and not request.get('pool_recovery_v1') and request.get('load_only') is False
            and request.get('production_released') is False and request.get('benchmark_credit') == 0,
            'pool recording requires exact bounded action scope')
    result = dict(request); result.pop('pool_active_recording_v1')
    if proof is not None:
        result.update(runtime_compatibility=proof['parent_compatibility'],
                      probe_runtime_sha256=proof['parent_runtime']['sha256'])
    return result


def verify_runtime(root, request, records, runtime, verify_parent):
    parent_request(request)
    reference = request['runtime_compatibility']; path = Path(reference['path']).resolve()
    require(path.is_relative_to(root) and file_digest(path) == reference['sha256'], 'recording proof changed')
    proof = json.loads(path.read_text()); parent_ref = proof['parent_runtime']
    parent = Path(parent_ref['path']).resolve().parent
    require(proof.get('schema') == SCHEMA and proof.get('physical_delta') == DELTA
            and proof.get('permitted_execution_kinds') == ['physics_action_open_loop']
            and parent.is_relative_to(root/'code') and runtime.is_relative_to(root/'code')
            and parent_ref == {'path': str(parent/'.snapshot_commit'), 'sha256': file_digest(parent/'.snapshot_commit')}
            and (parent/'.snapshot_commit').read_text().strip() == PARENT_COMMIT
            and proof['candidate_runtime'] == {'path': str(runtime/'.snapshot_commit'),
                                               'sha256': file_digest(runtime/'.snapshot_commit')},
            'pool recording runtime or scope differs')
    verify_parent(root, parent_request(request, proof), records, parent)
    paths = {str(p.relative_to(tree)) for tree in (parent, runtime) for folder in ('ispatialgym', 'scripts')
             for p in (tree/folder).rglob('*.py') if p.is_file()}
    changed = {p for p in paths if not (parent/p).is_file() or not (runtime/p).is_file()
               or file_digest(parent/p) != file_digest(runtime/p)}
    require(changed == set(proof['changed_payloads']) and changed <= ALLOWED
            and all((runtime/p).resolve().is_relative_to(runtime) and file_digest(runtime/p) == h
                    for p, h in proof['changed_payloads'].items()), 'unapproved recording payload')
    return reference


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
    names = serialized_object_names(state); inactive = set(initialized)-{active}
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
    # This native Scene.save wraps its whole loaded metadata in metadata.task,
    # then overlays the current binding and robot poses. Verify that exact
    # known wrapper before replacing it with the incoming source's metadata.
    # Unknown fields or references must never be silently discarded.
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
    require(set(target) == set(expected) == {'manifest_sha256', 'batch_id', 'active_names',
                                           'inactive_omitted', 'active_properties'}
            and all(target[k] == expected[k] for k in ('manifest_sha256', 'batch_id'))
            and expected['active_names'] == [active]
            and set(expected['active_properties']) == {active}
            and properties == expected['active_properties'][active]
            and active in bindings.values()
            and result['metadata']['task']['inst_to_name'] == bindings,
            'native scene metadata or current task binding differs')
    target.update(active_names=[active], inactive_omitted=len(initialized)-1,
                  active_properties={active: copy.deepcopy(properties)})
    require(result['init_info']['args']['scene_file'] == str(initial_scene_path),
            'native initialization scene differs from recorded environment')
    result['init_info']['args']['scene_file'] = str(exported_scene_path)
    return active_scene(result, initialized, {active})


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


def install(runner, contract, current, output, native_build):
    from .training_pool_consumers import recorded_registry_scope
    from .training_pool_adapter import source_member
    from ispatialgym.sampling.model_pool import native_properties
    initialized = set(contract['initialized_actors']); output = Path(output); observations = []
    known_metadata = [json.loads(Path(m['scene']['path']).read_text())['metadata'] for m in contract['members']]
    native_restore = runner.InstanceLoader.restore
    native_init = runner.Executor.__init__; native_dump = runner.Executor._dump_state
    native_start = runner.Recorder.start; native_snapshot = runner.Recorder._snapshot_scene

    def persist():
        write_json_atomic(output/'pool_recording_audit.json', {'physical_delta': DELTA,
            'observations': observations, 'benchmark_credit': 0, 'production_released': False,
            'renderer_equivalence_verified': False})

    def restored(loader, *args, **kwargs):
        result = native_restore(loader, *args, **kwargs)
        require(not loader.external_camera and not getattr(loader, 'behavior_output', False),
                'external/behavior camera configuration is outside this recording diagnostic')
        current['loader'] = loader
        current['recording_config'] = native_build(loader.template_dir, loader.tmeta, loader.tro,
            loader.tro_path, loader.activity, loader.did, loader.robot_name, loader.image_size,
            behavior_output=False, scene_crop=loader.scene_crop)
        return result

    def initialized_executor(executor, *args, **kwargs):
        native_init(executor, *args, **kwargs)
        require(executor.env is current['loader'].env and executor.recorder is not None,
                'recording executor does not own current pool world')
        executor.recorder._pool_recording_executor = executor

    def record_state(executor, *, initial):
        scene = executor.env.scene
        with recorded_registry_scope(scene, initialized, {current['active']}, initial_state=initial):
            # Direct owner dump for the explicit initial record. Full internal
            # snapshots always use the untouched native _dump_state path.
            flat = executor.state_owner.dump_state(serialized=True) if initial else native_dump(executor, False)
        evidence = verify_serialized(executor.state_owner, flat, initialized, current['active'],
                                     scene.object_registry.get_dict('name'), initial)
        evidence.update(ordinal=current['ordinal'], active=current['active'], tro=current['tro'])
        observations.append(evidence)
        return flat

    def dumped(executor, full):
        return native_dump(executor, full) if full else record_state(executor, initial=False)

    def started(recorder, obs0, info0=None, state0=None):
        require(state0 is not None and recorder.cfg.hdf5, 'active recording requires native initial state')
        result = native_start(recorder, obs0, info0, state0=record_state(recorder._pool_recording_executor, initial=True))
        require(isinstance(recorder.scene_file, dict) and isinstance(recorder.env_config, dict),
                'native active snapshot missing')
        persist(); return result

    def snapshot(recorder):
        native_snapshot(recorder)
        require(recorder.scene_file is not None and recorder.env_config is not None,
                'native scene snapshot failed')
        document = recorder.scene_file
        if isinstance(document, str):
            document = json.loads(document) if document.lstrip().startswith('{') else json.loads(Path(document).read_text())
        member = source_member(contract, current['loader'].tro_path)
        source = json.loads(Path(member['scene']['path']).read_text())
        bindings = {k: v.name for k, v in current['loader'].task.object_scope.items() if hasattr(v, 'name')}
        scene = recorder.env.scene; actor = scene.object_registry('name', current['active'])
        initial_scene = Path(recorder.env_config['scene']['scene_file']).resolve()
        require(initial_scene.parent == output and initial_scene.name in {
            'private_union_'+name+'.json' for name in initialized}, 'native union source is outside this diagnostic')
        destination = output/('recorded_active_scene_'+str(current['ordinal'])+'.json')
        projected = active_document(document, source, known_metadata, initialized, current['active'],
                                    native_properties(actor), bindings, current['loader'].tro.robot_poses,
                                    initial_scene, destination)
        require(not destination.exists(), 'preserve prior active recording scene')
        write_json_atomic(destination, projected)
        config = recorded_config(current['recording_config'], destination, member['tro']['path'],
                                 recorder.nominal_image_size, recorder.env_config)
        require(not any(name in json.dumps(config) for name in initialized-{current['active']}),
                'inactive actor referenced by recorded configuration')
        recorder.scene_file = projected; recorder.env_config = config
        current['active_recording_scene'] = {'path': str(destination), 'sha256': file_digest(destination)}

    runner.InstanceLoader.restore = restored
    runner.Executor.__init__ = initialized_executor; runner.Executor._dump_state = dumped
    runner.Recorder.start = started; runner.Recorder._snapshot_scene = snapshot
    return persist
