"""Bounded experiment clearing contact observations at an episode reset.

The native API deliberately preserves cached pairs across handle rebuilds.
Those pairs describe the preceding episode after actors have been teleported.
This experiment invalidates only those observations before the existing reset
step. It never changes collision filters, poses, forces, or physics timesteps.
"""
import copy
import json
from pathlib import Path

from .contracts import file_digest, require, write_json_atomic

SCHEMA = 'isg-reset-contact-cache-runtime-compatibility-v1'
PARENT_COMMIT = 'afb8050c68fc25749c798294a95e0c90fec9caef'
DELTA = {'name': 'reset_contact_observations_before_existing_step_v1',
         'boundary': 'task_load_scene_reset_after_state_restore_before_existing_physics_step',
         'cold_physical_operations_changed': True, 'warm_physical_operations_changed': True,
         'additional_physics_steps': 0, 'source_inputs_changed': False,
         'collision_filters_changed': False, 'thresholds_changed': False,
         'native_contact_views_rebuilt': False}
ALLOWED = {'scripts/campaign_20261110/training_contact_reset_probe.py',
           'scripts/campaign_20261110/tests/test_training_contact_reset_probe.py',
           'scripts/campaign_20261110/training_batch_probe.py'}
MATRICES = ('_CONTACT_MATRIX', '_CURRENT_CONTACT_MATRIX',
            '_CONTACT_MATRIX_GPU', '_CURRENT_CONTACT_MATRIX_GPU')


def verify_runtime(root, request, records, runtime, verify_parent):
    require(request.get('reset_contact_cache_before_step_v1') is True
            and request.get('fixed_mass_baseline') is True
            and request.get('production_released') is False and request.get('benchmark_credit') == 0,
            'contact reset requires a bounded fixed-mass diagnostic')
    reference = request['runtime_compatibility']; path = Path(reference['path']).resolve()
    require(path.is_relative_to(root) and file_digest(path) == reference['sha256'],
            'contact reset proof changed')
    proof = json.loads(path.read_text()); parent_ref = proof['parent_runtime']
    parent = Path(parent_ref['path']).resolve().parent
    require(proof.get('schema') == SCHEMA and proof.get('physical_delta') == DELTA
            and proof.get('permitted_execution_kinds') == ['physics_action_open_loop']
            and proof.get('physical_code_changes') == 1,
            'contact reset delta or execution kind differs')
    require(parent.is_relative_to(root/'code') and runtime.is_relative_to(root/'code')
            and parent_ref == {'path': str(parent/'.snapshot_commit'),
                               'sha256': file_digest(parent/'.snapshot_commit')}
            and (parent/'.snapshot_commit').read_text().strip() == PARENT_COMMIT
            and proof['candidate_runtime'] == {'path': str(runtime/'.snapshot_commit'),
                                               'sha256': file_digest(runtime/'.snapshot_commit')},
            'contact reset runtime identity differs')
    inherited = {**request, 'runtime_compatibility': proof['parent_compatibility'],
                 'probe_runtime_sha256': parent_ref['sha256']}
    inherited.pop('reset_contact_cache_before_step_v1')
    verify_parent(root, inherited, records, parent)
    paths = {str(p.relative_to(tree)) for tree in (parent, runtime)
             for folder in ('ispatialgym', 'scripts') for p in (tree/folder).rglob('*.py') if p.is_file()}
    changed = {n for n in paths if not (parent/n).is_file() or not (runtime/n).is_file()
               or file_digest(parent/n) != file_digest(runtime/n)}
    require(changed == set(proof['changed_payloads']) and changed <= ALLOWED
            and all((runtime/n).resolve().is_relative_to(runtime) and file_digest(runtime/n) == h
                    for n, h in proof['changed_payloads'].items()),
            'unapproved contact reset payload')
    return reference


def clear_observations(api, scene_index, *, synchronize=lambda: None):
    """Clear CPU/GPU aliases in place; preserve views, maps and graph storage."""
    require(api._PENDING_STEPS == 0, 'contact reset would discard unconsumed physics observations')
    tensors = []
    for name in MATRICES:
        table = getattr(api, name, None)
        require(isinstance(table, dict) and scene_index in table, 'contact cache is unavailable: '+name)
        value = table[scene_index]
        require(len(value.shape) == 2 and value.numel() > 0 and str(value.dtype) == 'torch.bool',
                'contact cache layout differs: '+name)
        tensors.append((name, value))
    require(len({tuple(t.shape) for _, t in tensors}) == 1, 'contact cache mirror shapes differ')
    synchronize()
    before = {name: int(value.count_nonzero().item()) for name, value in tensors}
    pointers = {name: value.data_ptr() for name, value in tensors}
    for _, value in tensors:
        value.zero_()
    synchronize()
    after = {name: int(value.count_nonzero().item()) for name, value in tensors}
    require(not any(after.values()) and all(value.data_ptr() == pointers[name] for name, value in tensors),
            'contact cache reset failed or replaced graph storage')
    return {'scene_index': scene_index, 'pairs_before': before, 'pairs_after': after,
            'pending_steps': 0, 'storage_preserved': True, 'physics_steps_added': 0}


def around_one_reset_step(scene, sim, load, before_step):
    """Intercept only the reset called by this task load; preserve native calls."""
    reset = scene.reset; had_reset = 'reset' in vars(scene); raw_reset = vars(scene).get('reset')
    calls = []
    def observed_reset(*args, **kwargs):
        step = sim.step_physics; had_step = 'step_physics' in vars(sim); raw_step = vars(sim).get('step_physics')
        def observed_step(*step_args, **step_kwargs):
            require(not calls, 'task load requested more than one reset physics step')
            calls.append(before_step())
            return step(*step_args, **step_kwargs)
        sim.step_physics = observed_step
        try:
            return reset(*args, **kwargs)
        finally:
            if had_step: sim.step_physics = raw_step
            else: del sim.step_physics
    scene.reset = observed_reset
    try:
        result = load()
    finally:
        if had_reset: scene.reset = raw_reset
        else: del scene.reset
    require(len(calls) == 1, 'contact diagnostic did not observe one existing reset step')
    return result, calls[0]


def install(runner, request, output):
    require(request.get('reset_contact_cache_before_step_v1') is True
            and request.get('production_released') is False,
            'contact reset experiment not requested')
    import torch
    from omnigibson.utils.usd_utils import RigidContactAPI
    restore = runner.InstanceLoader.restore; events = []; output = Path(output)
    def restored(loader, *args, **kwargs):
        task = loader.task; load = task.load_tro_instance
        had_load = 'load_tro_instance' in vars(task); raw_load = vars(task).get('load_tro_instance')
        def observed_load(*load_args, **load_kwargs):
            require(not runner.og.sim.currently_stepping, 'reset diagnostic called during physics')
            scene_index = runner.og.sim.scenes.index(loader.env.scene)
            def before():
                evidence = clear_observations(RigidContactAPI, scene_index,
                                              synchronize=torch.cuda.synchronize)
                event = {'instance_id': loader.iid, 'source_tro': {'path': str(loader.tro_path),
                         'sha256': file_digest(loader.tro_path)}, **evidence}
                events.append(event)
                write_json_atomic(output/'contact_reset_audit.json', {
                    'schema': 'isg-reset-contact-observations-audit-v1', 'physical_delta': DELTA,
                    'runtime_compatibility': request['runtime_compatibility'], 'events': events,
                    'benchmark_credit': 0, 'production_released': False})
                return event
            result, event = around_one_reset_step(loader.env.scene, runner.og.sim,
                lambda: load(*load_args, **load_kwargs), before)
            loader.report['contact_reset_diagnostic'] = copy.deepcopy(event)
            return result
        task.load_tro_instance = observed_load
        try:
            return restore(loader, *args, **kwargs)
        finally:
            if had_load: task.load_tro_instance = raw_load
            else: del task.load_tro_instance
    runner.InstanceLoader.restore = restored
