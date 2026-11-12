"""Use the same exact source anchor operation in cold and reused pool worlds.

No reference observation is copied into physics. Both paths use the original
incoming TRO spawn through the native stopped robot setter, including its fixed
world joint. The measured zero anchor offset is a required scope restriction.
"""
import copy
import inspect
import json
import math
import os
from pathlib import Path

from .contracts import require, file_digest, write_json_atomic

SCHEMA = 'isg-pool-source-anchor-runtime-compatibility-v1'
PARENT_COMMIT = '7896730876bc7c0c3432d6e914f0354802eb2b21'
DELTA = {'name': 'pool_exact_source_anchor_v1', 'source_inputs_changed': False,
         'cold_boundary': 'existing_evaluation_mass_stopped_callback',
         'warm_boundary': 'existing_pool_attach_stopped_callback',
         'anchor_input': 'exact_incoming_TRO_spawn_without_SE3_roundoff_composition',
         'world_fixed_joint_set_by_native_robot_setter': True,
         'extra_explicit_physics_steps': 0, 'extra_stop_play_pairs': 0,
         'pool_activation_operations_changed': False, 'thresholds_changed': False,
         'reference_state_copied': False}
ALLOWED = {'scripts/campaign_20261110/training_pool_source_anchor_probe.py',
           'scripts/campaign_20261110/tests/test_training_pool_source_anchor_probe.py',
           'scripts/campaign_20261110/training_pool_probe.py'}


def verify_runtime(root, request, records, runtime, verify_parent):
    require(request.get('pool_exact_source_anchor_v1') is True and request.get('selected_pool_world') is True
            and request.get('load_only') is True and request.get('production_released') is False
            and request.get('benchmark_credit') == 0, 'source anchor requires bounded pool load scope')
    reference = request['runtime_compatibility']; path = Path(reference['path']).resolve()
    require(path.is_relative_to(root) and file_digest(path) == reference['sha256'], 'source anchor proof changed')
    proof = json.loads(path.read_text()); parent_ref = proof['parent_runtime']
    parent = Path(parent_ref['path']).resolve().parent
    require(proof.get('schema') == SCHEMA and proof.get('physical_delta') == DELTA
            and proof.get('permitted_execution_kinds') == ['native_load_only']
            and parent.is_relative_to(root/'code') and runtime.is_relative_to(root/'code')
            and parent_ref == {'path': str(parent/'.snapshot_commit'), 'sha256': file_digest(parent/'.snapshot_commit')}
            and (parent/'.snapshot_commit').read_text().strip() == PARENT_COMMIT
            and proof['candidate_runtime'] == {'path': str(runtime/'.snapshot_commit'),
                                               'sha256': file_digest(runtime/'.snapshot_commit')},
            'source anchor runtime or physical scope differs')
    native = runtime/'Behavior-1K/OmniGibson/omnigibson/robots/robot.py'
    require(proof['native_robot_setter']['sha256'] == file_digest(native), 'native robot setter differs')
    inherited = {**request, 'runtime_compatibility': proof['parent_compatibility'],
                 'probe_runtime_sha256': parent_ref['sha256']}
    inherited.pop('pool_exact_source_anchor_v1')
    verify_parent(root, inherited, records, parent)
    paths = {str(p.relative_to(tree)) for tree in (parent, runtime) for folder in ('ispatialgym', 'scripts')
             for p in (tree/folder).rglob('*.py') if p.is_file()}
    changed = {p for p in paths if not (parent/p).is_file() or not (runtime/p).is_file()
               or file_digest(parent/p) != file_digest(runtime/p)}
    require(changed == set(proof['changed_payloads']) and changed <= ALLOWED
            and all((runtime/p).resolve().is_relative_to(runtime) and file_digest(runtime/p) == h
                    for p, h in proof['changed_payloads'].items()), 'unapproved source anchor payload')
    return reference


def source_anchor(baseline, incoming):
    first = baseline['spawn_pose']
    require(baseline['anchor_position'] == first['position'], 'nonzero anchor translation needs a separate contract')
    q = baseline['anchor_orientation']; expected = first['orientation']
    require(min(math.dist(q, expected), math.dist(q, [-v for v in expected])) <= 1e-6,
            'nonzero anchor rotation needs a separate contract')
    require(set(incoming) >= {'position', 'orientation'} and len(incoming['position']) == 3
            and len(incoming['orientation']) == 4
            and all(math.isfinite(float(v)) for key in ('position', 'orientation') for v in incoming[key])
            and abs(math.sqrt(sum(float(v)**2 for v in incoming['orientation']))-1) <= 1e-6,
            'source anchor pose is invalid')
    return {key: copy.deepcopy(incoming[key]) for key in ('position', 'orientation')}


def observed_no_steps(sim, call):
    calls = []; original = []
    before = (sim.get_physics_dt(), sim.get_rendering_dt(), sim.get_sim_step_dt())
    for obj, name in ((sim, 'step_physics'), (sim._sim_context, 'step')):
        native = getattr(obj, name); had = name in vars(obj); old = vars(obj).get(name)
        def recorded(*args, _native=native, _name=name, **kwargs):
            calls.append(_name); return _native(*args, **kwargs)
        original.append((obj, name, had, old)); setattr(obj, name, recorded)
    try:
        result = call()
    finally:
        for obj, name, had, old in reversed(original):
            if had: setattr(obj, name, old)
            else: delattr(obj, name)
    require(not calls and before == (sim.get_physics_dt(), sim.get_rendering_dt(), sim.get_sim_step_dt()),
            'source anchor lifecycle changed physics stepping or timestep')
    return result, {'observed_physics_step_calls': calls, 'time_step_settings': list(before)}


def install(runner, contract, output, parent_install, request):
    from . import training_pool_probe as pool
    parent_install(runner, contract, output)
    build = runner.InstanceLoader.build_env; attach = runner.InstanceLoader.attach
    events = []; output = Path(output)
    proof = json.loads(Path(request['runtime_compatibility']['path']).read_text())
    def persist():
        write_json_atomic(output/'source_anchor_audit.json', {'physical_delta': DELTA, 'events': events,
            'benchmark_credit': 0, 'production_released': False})
    pool.anchor_for_member = source_anchor
    def cold(loader):
        original = runner.M.apply_eval_base_mass
        observations = []
        def with_anchor(robot, mass_kg=None, while_stopped=None):
            require(robot.is_holonomic_base and
                    file_digest(inspect.getfile(robot.set_position_orientation)) == proof['native_robot_setter']['sha256'],
                    'source anchor native robot implementation differs')
            target_mass = float(os.environ.get('ISG_BASE_MASS_KG', runner.M.EVAL_BASE_LINK_MASS_KG)) if mass_kg is None else float(mass_kg)
            require(target_mass > 0 and abs(float(robot.base_footprint_link.mass)-target_mass) >= 1e-6,
                    'cold anchor may only join an already-required mass lifecycle')
            incoming = pool._incoming_spawn(loader, robot)
            def stopped():
                require(runner.og.sim.is_stopped(), 'native cold anchor callback was not stopped')
                if while_stopped is not None: while_stopped()
                robot.set_position_orientation(position=incoming['position'], orientation=incoming['orientation'])
            result, steps = observed_no_steps(runner.og.sim, lambda:
                original(robot, mass_kg=mass_kg, while_stopped=stopped))
            observations.append(steps)
            return result
        runner.M.apply_eval_base_mass = with_anchor
        try:
            result = build(loader)
        finally:
            runner.M.apply_eval_base_mass = original
        require(len(observations) == 1, 'cold source anchor did not use exactly one existing mass lifecycle')
        events.append({'phase': 'cold_after_existing_mass_lifecycle', 'tro': loader.tro_path,
            'source_spawn': pool._incoming_spawn(loader, loader.robot),
            'native_anchor': [v.tolist() for v in loader.robot.root_link.get_position_orientation()],
            **observations[0]}); persist()
        return result
    def warm(loader, shared):
        result = attach(loader, shared)
        events.append({'phase': 'warm_after_existing_anchor_lifecycle', 'tro': loader.tro_path,
            'source_spawn': pool._incoming_spawn(loader, loader.robot),
            'native_anchor': [v.tolist() for v in loader.robot.root_link.get_position_orientation()]}); persist()
        return result
    runner.InstanceLoader.build_env = cold
    runner.InstanceLoader.attach = warm
