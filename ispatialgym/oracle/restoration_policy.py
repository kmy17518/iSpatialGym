"""Explicit source/runtime-bound shared-loader restoration policy.

Ordinary restoration and explicitly counted native reconstruction are separate
options. Existing production-purpose input authorization runs independently;
this bounded policy does not grant a production release.
"""
import json
from contextlib import contextmanager
from pathlib import Path

from scripts.campaign_20261110.contracts import file_digest, require

OPTIONS = {'schema': 'isg-shared-loader-restoration-options-v1',
           'initialized_fixed_frames': 'all_eligible_initialized_fixed_kinematic',
           'complete_native_snapshots': True, 'physics_reconstruction': 'none'}
BACKGROUND_ROUNDTRIP_OPTIONS = {**OPTIONS, 'physics_reconstruction': 'late_incoming_v1',
                               'serialized_background_roots': 'exact_two_cycle_v1'}
BACKGROUND_INVERSE_OPTIONS = {**BACKGROUND_ROUNDTRIP_OPTIONS,
                             'serialized_background_roots': 'exact_bounded_inverse_v2'}
SOURCE_MASS_OPTIONS = {**OPTIONS, 'physics_reconstruction':'late_incoming_v1',
                       'initial_root_mass_baseline':'serialized_root_constraints_v1'}


def checked_options(options):
    require(options in (OPTIONS, {**OPTIONS, 'physics_reconstruction': 'late_incoming_v1'},
                        BACKGROUND_ROUNDTRIP_OPTIONS, BACKGROUND_INVERSE_OPTIONS, SOURCE_MASS_OPTIONS),
            'unmeasured shared-loader restoration policy')
    return dict(options)


def _reference(path): return {'path': str(path), 'sha256': file_digest(path)}


def bind(root, request_ref, source_ref, runtime):
    root = Path(root).resolve(); runtime = Path(runtime).resolve()
    def checked(reference):
        path = Path(reference['path']).resolve()
        require(path.is_relative_to(root) and _reference(path) == reference,
                'shared-loader policy artifact changed or escaped campaign')
        return path
    request_path = checked(request_ref); source_path = checked(source_ref)
    q = json.loads(request_path.read_text()); options = checked_options(q['shared_loader_restoration_policy'])
    rebuilds = int(options['physics_reconstruction'] == 'late_incoming_v1')
    require(q['physical_delta'].get('shared_loader_restoration_policy') == options
            and q['physical_delta'].get('explicit_sdk_reconstructions') == rebuilds
            and q['physical_delta'].get('additional_stop_play_pairs') == rebuilds,
            'shared-loader declared physical route differs from its options')
    require(q['schema'] == 'isg-retained-mechanism-replay-request-v1'
            and q['campaign_id'] == root.name == 'campaign_20261110'
            and q['production_released'] is False and type(q['benchmark_credit']) is int
            and q['benchmark_credit'] == 0, 'shared-loader policy requires bounded zero-credit execution')
    bindings = q['shared_loader_source_tros']
    require(isinstance(bindings, list) and 0 < len(bindings) <= 24
            and len({v['path'] for v in bindings}) == len(bindings) and source_ref in bindings,
            'shared-loader policy does not bind this exact source TRO')
    for reference in bindings: checked(reference)
    raw = json.loads(source_path.read_text())['ispatialgym']
    require(raw['instance_id'] == q['instance_id'], 'shared-loader source identity differs')
    manifest = json.loads(checked(q['source_manifest']).read_text())
    source_rows = [row for row in manifest['rows'] if row['instance_id'] == q['instance_id']]
    require(len(source_rows) == 1 and
            {k: source_rows[0]['source_tro'][k] for k in ('path', 'sha256')} == source_ref,
            'shared-loader policy source differs from its immutable source manifest')
    marker = runtime/'.snapshot_commit'
    require(runtime.is_relative_to(root/'code') and _reference(marker) == q['native_runtime'],
            'shared-loader policy native runtime differs')
    proof_ref = q['runtime_proof']; proof = json.loads(checked(proof_ref).read_text())
    require(proof['candidate_runtime'] == q['native_runtime']
            and proof['source_manifest'] == q['source_manifest']
            and proof['physical_delta'] == q['physical_delta']
            and proof['production_released'] is False,
            'shared-loader runtime proof differs from the exact request')
    if options['physics_reconstruction'] != 'none':
        for key in ('installed_api_bindings', 'installed_api_definition', 'installed_simulation_route_test'):
            item = proof.get(key)
            require(isinstance(item, dict) and Path(item['path']).is_absolute()
                    and file_digest(item['path']) == item['sha256'],
                    'shared-loader inspected native reconstruction API changed')
    payloads = {str(p.relative_to(runtime)): file_digest(p)
                for folder in ('ispatialgym', 'scripts')
                for p in (runtime/folder).rglob('*.py') if p.is_file()}
    require(payloads == proof['candidate_python_payloads'],
            'shared-loader candidate Python payload changed')
    if q.get('source_authorization') is not None:
        from .height_source_authorization import authorize
        authorize(root, q, source_ref, runtime)
    bound = {'schema': 'isg-bound-shared-loader-restoration-policy-v1', 'root': str(root),
            'request': dict(request_ref), 'source_tro': dict(source_ref),
            'native_runtime': dict(q['native_runtime']), 'runtime_proof': dict(proof_ref),
            'options': options, 'production_released': False, 'benchmark_credit': 0}
    if q.get('source_authorization') is not None:
        bound['source_authorization'] = dict(q['source_authorization'])
    return bound


def validate(policy, tro_path, runtime):
    if policy is None: return None
    require(policy['schema'] == 'isg-bound-shared-loader-restoration-policy-v1'
            and _reference(Path(tro_path).resolve()) == policy['source_tro'],
            'shared-loader policy source binding differs')
    actual = bind(policy['root'], policy['request'], policy['source_tro'], runtime)
    require(actual == policy, 'shared-loader policy changed after source validation')
    return actual


def capture(loader):
    if loader._restoration_policy is None: return
    if loader._restoration_policy['options'].get('initial_root_mass_baseline') is not None:
        from .source_mass_baseline import initialize
        import omnigibson as og
        initialize(loader, og.sim)
    from . import initialized_fixed_frames as frames
    require(not hasattr(loader, '_initialized_fixed_frame_baseline'),
            'shared loader cannot recapture initialized frames from a used world')
    loader._initialized_fixed_frame_baseline = frames.capture(loader.env.scene)
    loader.report['shared_loader_restoration_policy'] = loader._restoration_policy
    loader.report['initialized_fixed_frame_capture'] = frames.plain(loader._initialized_fixed_frame_baseline)


def share(loader, shared):
    if loader._restoration_policy is None: return
    shared['initialized_fixed_frame_baseline'] = loader._initialized_fixed_frame_baseline
    shared['shared_loader_restoration_options'] = loader._restoration_policy['options']
    shared['shared_loader_restoration_runtime'] = loader._restoration_policy['native_runtime']
    if hasattr(loader, '_source_mass_baseline'):
        shared['source_root_mass_baseline'] = loader._source_mass_baseline


def attach(loader, shared):
    if loader._restoration_policy is None:
        require('shared_loader_restoration_options' not in shared,
                'shared world cannot silently drop its restoration policy')
        return
    require(shared.get('shared_loader_restoration_options') == loader._restoration_policy['options']
            and shared.get('shared_loader_restoration_runtime') == loader._restoration_policy['native_runtime']
            and shared.get('initialized_fixed_frame_baseline') is not None,
            'shared world restoration policy or initialized baseline differs')
    loader._initialized_fixed_frame_baseline = shared['initialized_fixed_frame_baseline']
    if loader._restoration_policy['options'].get('initial_root_mass_baseline') is not None:
        require(shared.get('source_root_mass_baseline') is not None,
                'shared source mass initialization baseline is missing')
        loader._source_mass_baseline = shared['source_root_mass_baseline']
    from .height_source_authorization import before_mutation
    before_mutation(loader)
    if hasattr(loader, '_source_mass_baseline'):
        from .source_mass_baseline import verify
        verify(loader, 'before_attach_state_restore')


def restore(loader, sim):
    if loader._restoration_policy is None: return
    from . import initialized_fixed_frames as frames
    loader.report['shared_loader_restoration_policy'] = loader._restoration_policy
    from .height_source_authorization import before_mutation
    before_mutation(loader)
    if hasattr(loader, '_source_mass_baseline'):
        from .source_mass_baseline import verify
        verify(loader, 'before_episode_restore')
    event = {}; loader.report['initialized_fixed_frame_restore'] = event
    frames.restore(loader.env.scene, loader._initialized_fixed_frame_baseline, sim, event=event)


def restore_background_roots(loader, state, sim, *, phase):
    policy = loader._restoration_policy
    if policy is None or policy['options'].get('serialized_background_roots') is None:
        return
    from .height_source_authorization import before_mutation
    from .native_root_roundtrip import restore as restore_roots
    before_mutation(loader)
    excluded = {loader.robot.name}
    excluded.update(value['scene_object_name'] if isinstance(value, dict) else value
                    for value in loader.tro.meta.object_bindings.values())
    event = {'phase': phase}
    loader.report.setdefault('serialized_background_root_restore', []).append(event)
    max_writes = (2 if policy['options']['serialized_background_roots'] == 'exact_bounded_inverse_v2'
                  and phase == 'after_attach_state_restore' else 1)
    restore_roots(loader.env.scene, state, excluded, sim, event=event, max_writes=max_writes)


def needs_reconstruction(loader):
    return (loader._restoration_policy is not None and
            loader._restoration_policy['options']['physics_reconstruction'] == 'late_incoming_v1')


@contextmanager
def before_native_reset_step(sim, callback, event):
    """Intercept only the existing reset step, after its native state load."""
    original = sim.step_physics; had = 'step_physics' in vars(sim)
    prior = vars(sim).get('step_physics'); event['existing_reset_steps_observed'] = 0
    def stepping(*args, **kwargs):
        require(event['existing_reset_steps_observed'] == 0,
                'shared reconstruction requires exactly one native reset step')
        event['existing_reset_steps_observed'] += 1
        callback()
        return original(*args, **kwargs)
    sim.step_physics = stepping
    try:
        yield
        require(event['existing_reset_steps_observed'] == 1,
                'shared reconstruction did not observe its native reset step')
    finally:
        if had: sim.step_physics = prior
        else: del sim.step_physics


@contextmanager
def reconstruction_reset_scope(loader, sim):
    require(needs_reconstruction(loader), 'shared native reconstruction was not authorized')
    from omni.physx import get_physx_interface, get_physx_simulation_interface
    from omni.physx.bindings import _physx
    from .campaign_restore import restart_loaded_physics
    from .native_reconstruction import reconstruct
    policy = loader._restoration_policy
    require(file_digest(policy['runtime_proof']['path']) == policy['runtime_proof']['sha256'],
            'shared-loader reconstruction proof changed after binding')
    proof = json.loads(Path(policy['runtime_proof']['path']).read_text())
    require(file_digest(_physx.__file__) == proof['installed_api_bindings']['sha256'],
            'shared-loader loaded SDK bindings changed')
    event = {'status': 'started', 'instance_id': loader.iid,
             'boundary': 'incoming_state_loaded_immediately_before_existing_first_reset_physics_step'}
    loader.report.setdefault('shared_native_reconstruction', []).append(event)
    def restart(meta, simulation, *, execution, while_stopped):
        require(meta == loader.tro.meta.raw['campaign'] and execution == loader._restoration_execution,
                'shared native reconstruction source authorization changed')
        value = restart_loaded_physics(simulation, while_stopped=while_stopped)
        value['boundary'] = event['boundary']; return value
    def callback():
        from .height_source_authorization import before_mutation
        authorization = before_mutation(loader)
        if authorization is not None:
            event['original_canary_execution'] = authorization
        extra = ({'after_restore': lambda state: restore_background_roots(
            loader, state, sim, phase='after_own_sdk_state_restore')}
            if policy['options'].get('serialized_background_roots') is not None else {})
        reconstruct(loader, sim, restart, get_physx_interface(), get_physx_simulation_interface(), event, **extra)
    try:
        with before_native_reset_step(sim, callback, event): yield
        event['status'] = 'completed'
    except BaseException as error:
        event.update(status='failed', error=repr(error)); raise


@contextmanager
def complete_snapshot_scope(loader):
    """Observe every actual native snapshot while the full registry is selected."""
    from .campaign_restore import native_snapshot_scope
    scene = loader.env.scene; original = scene.update_initial_file
    had = 'update_initial_file' in vars(scene); prior = vars(scene).get('update_initial_file')
    events = []; loader.report['complete_native_snapshot_inventory'] = events
    def updated(*args, **kwargs):
        answer = original(*args, **kwargs)
        names = sorted(scene._initial_file['state']['registry']['object_registry'])
        native = sorted(obj.name for obj in scene.objects)
        event = {'snapshot_objects': names, 'native_objects': native,
                 'complete': names == native, 'physics_steps_added': 0}
        events.append(event); require(event['complete'], 'shared loader snapshot omitted initialized actors')
        return answer
    with native_snapshot_scope(scene, loader.report):
        scene.update_initial_file = updated
        try:
            yield
            require(events, 'shared loader did not publish a native snapshot')
            if hasattr(loader, '_source_mass_baseline'):
                from .source_mass_baseline import verify
                verify(loader, 'after_episode_restore')
        finally:
            if had: scene.update_initial_file = prior
            else: del scene.update_initial_file
