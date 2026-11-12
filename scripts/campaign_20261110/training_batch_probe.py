"""Execute an observational A/B/A probe through existing native production paths.

Inputs are separate validated canaries. Outputs never allocate benchmark quota.
A modern CPU helper must re-audit the resulting evidence before any release.
"""
import argparse
import ast
from contextlib import contextmanager
import json
import os
from pathlib import Path
from scripts.campaign_20261110.contracts import file_digest,require,write_json_atomic


def verify_saved_checker_delta(base, runtime, proof):
    """The reviewed checker may change metadata validation, never state restoration."""
    relative='ispatialgym/oracle/campaign_batch.py'
    binding=proof.get('saved_scene_checker') or {}
    require(binding.get('base_sha256')==file_digest(base/relative) and
            binding.get('runtime_sha256')==file_digest(runtime/relative),
            'saved-scene checker payload differs')
    def physical(path):
        tree=ast.parse(path.read_text())
        tree.body=[node for node in tree.body if not
            (isinstance(node,ast.FunctionDef) and node.name in
             ('checked_nested_parking_provenance','saved_world_scope'))]
        return ast.dump(tree,include_attributes=False)
    require(physical(base/relative)==physical(runtime/relative),
            'saved-scene checker changed restoration or other physical operations')


def verify_runtime_compatibility(root, request, records, runtime):
    """Recheck the narrow observation-only runtime against native source bytes."""
    root=Path(root).resolve();runtime=Path(runtime).resolve()
    if request.get('selected_pool_world') is True:
        from .training_pool_probe import verify_runtime
        return verify_runtime(root, request, records, runtime)
    if request.get('physical_candidate'):
        from .training_pristine_probe import verify_candidate
        return verify_candidate(root, request, records, runtime)
    ref=request.get('runtime_compatibility') or {}
    proof_path=Path(ref.get('path','')).resolve()
    require(proof_path.is_relative_to(root) and proof_path.is_file() and
            file_digest(proof_path)==ref.get('sha256'),'missing exact observational runtime proof')
    proof=json.loads(proof_path.read_text())
    if proof.get('schema') == 'isg-physics-sdk-rebuild-runtime-compatibility-v1':
        from .training_physics_rebuild_probe import verify_runtime
        return verify_runtime(root, request, records, runtime, verify_runtime_compatibility)
    if proof.get('schema') == 'isg-reset-contact-cache-runtime-compatibility-v1':
        from .training_contact_reset_probe import verify_runtime
        return verify_runtime(root, request, records, runtime, verify_runtime_compatibility)
    if proof.get('schema') == 'isg-fixed-mass-baseline-runtime-compatibility-v1':
        from .training_fixed_mass_probe import verify_fixed_mass_runtime
        return verify_fixed_mass_runtime(root,request,records,runtime,verify_runtime_compatibility)
    if proof.get('schema') == 'isg-sequential-mass-restart-runtime-compatibility-v1':
        from .training_mass_restart_probe import verify_mass_restart_runtime
        return verify_mass_restart_runtime(root,request,records,runtime,verify_runtime_compatibility)
    if proof.get('schema') == 'isg-sequential-anchor-runtime-compatibility-v1':
        from .training_anchor_probe import verify_anchor_runtime
        return verify_anchor_runtime(root,request,records,runtime,verify_runtime_compatibility)
    if proof.get('schema') == 'isg-sequential-checker-runtime-compatibility-v1':
        require(request.get('sequential_checker_only') is True and request.get('sequential_augmentation') is True and
                request.get('production_released') is False and
                proof.get('permitted_execution_kinds')==['physics_action_open_loop'],
                'saved-scene checker requires exact bounded physics probe scope')
        base=Path(proof['native_runtime']['path']).resolve().parent
        require(base.is_relative_to(root/'code') and runtime.is_relative_to(root/'code') and
                file_digest(base/'.snapshot_commit')==proof['native_runtime']['sha256'] and
                proof['checker_runtime']=={'path':str(runtime/'.snapshot_commit'),'sha256':file_digest(runtime/'.snapshot_commit')} and
                all(r['validation_key']['code_sha256']==proof['native_runtime']['sha256']for r in records),
                'saved-scene checker source runtime differs')
        allowed={'ispatialgym/oracle/campaign_batch.py','ispatialgym/tests/test_campaign_batch.py',
                 'scripts/campaign_20261110/training_batch_probe.py',
                 'scripts/campaign_20261110/tests/test_training_batch_checker_compatibility.py'}
        paths={str(p.relative_to(tree))for tree in (base,runtime)for folder in ('ispatialgym','scripts')for p in(tree/folder).rglob('*.py')}
        changed={name for name in paths if not(base/name).is_file() or not(runtime/name).is_file() or file_digest(base/name)!=file_digest(runtime/name)}
        require(changed==set(proof['changed_payloads']) and changed<=allowed and proof.get('physical_code_changes')==0,
                'saved-scene checker changed native physics or unapproved code')
        require(all(file_digest(runtime/name)==sha for name,sha in proof['changed_payloads'].items()),'saved-scene checker payload changed')
        verify_saved_checker_delta(base,runtime,proof)
        return {'path':str(proof_path),'sha256':ref['sha256']}
    if proof.get('schema') == 'isg-render-cleanup-runtime-compatibility-v1':
        cleanup = request.get('render_cleanup_only') is True and request.get('diagnostic_control') in ('RENDER_FAILURE_A', 'RENDER_INTERRUPT_A')
        sequential = (request.get('sequential_render_only') is True and request.get('render_cleanup_only') is not True and
                      request.get('sequential_augmentation') is True and request.get('diagnostic_control') in (None,'FRESH_A','FRESH_B') and
                      request.get('production_released') is False and
                      proof.get('permitted_execution_kinds') == ['kinematic_observation_rendering'])
        require(cleanup or sequential, 'render cleanup runtime requires an exact render-only scope')
        base=Path(proof['native_runtime']['path']).resolve().parent
        require(base.is_relative_to(root/'code') and runtime.is_relative_to(root/'code') and
                file_digest(base/'.snapshot_commit')==proof['native_runtime']['sha256'] and
                file_digest(runtime/'.snapshot_commit')==proof['cleanup_runtime']['sha256'] and
                Path(proof['cleanup_runtime']['path']).resolve()==runtime/'.snapshot_commit' and
                all(r['validation_key']['code_sha256']==proof['native_runtime']['sha256']for r in records),
                'render cleanup runtime/source identity differs')
        allowed={'ispatialgym/multienv/replay.py','ispatialgym/multienv/campaign_render_cleanup.py',
                 'ispatialgym/multienv/campaign_rendering.py','scripts/campaign_20261110/training_batch_probe.py',
                 'ispatialgym/tests/test_campaign_render_cleanup.py',
                 'scripts/campaign_20261110/tests/test_training_render_cleanup_probe.py'}
        if proof.get('saved_scene_checker'):
            verify_saved_checker_delta(base,runtime,proof)
            allowed.update(('ispatialgym/oracle/campaign_batch.py','ispatialgym/tests/test_campaign_batch.py',
                            'scripts/campaign_20261110/tests/test_training_batch_checker_compatibility.py'))
        changed=set()
        for folder in ('ispatialgym','scripts'):
            paths={str(p.relative_to(tree))for tree in (base,runtime)for p in(tree/folder).rglob('*.py')}
            for relative in paths:
                a,b=base/relative,runtime/relative
                if not a.is_file()or not b.is_file()or file_digest(a)!=file_digest(b):changed.add(relative)
        require(changed==set(proof['changed_payloads']) and changed<=allowed and
                proof.get('physical_code_changes')==0,'render cleanup changed native physics or other code')
        require(all(file_digest(runtime/name)==sha for name,sha in proof['changed_payloads'].items()),
                'render cleanup payload changed')
        return {'path':str(proof_path),'sha256':ref['sha256']}
    if proof.get('schema') == 'isg-exact-native-runtime-compatibility-v1':
        stamp = runtime/'.snapshot_commit'
        require(runtime.is_relative_to(root/'code') and proof.get('native_runtime') ==
                {'path':str(stamp), 'sha256':file_digest(stamp)} and
                all(r['validation_key']['code_sha256'] == file_digest(stamp) for r in records),
                'exact observation runtime differs from native source')
        payloads = proof.get('python_payloads')
        paths = {str(p.relative_to(runtime)): p for folder in ('ispatialgym', 'scripts')
                 for p in (runtime/folder).rglob('*.py') if p.is_file()}
        require(proof.get('physical_code_changes') == 0 and isinstance(payloads, dict) and
                payloads and set(payloads) == set(paths), 'exact native payload inventory differs')
        require(all(p.resolve().is_relative_to(runtime) and file_digest(p) == payloads[name]
                    for name,p in paths.items()), 'exact native payload changed')
        return {'path':str(proof_path), 'sha256':ref['sha256']}
    require(proof.get('schema')=='isg-observational-runtime-compatibility-v1', 'invalid observational runtime proof')
    base=root/'code'/('trunk_'+proof['base_commit'][:12])
    require(runtime.is_relative_to(root/'code') and
            (runtime/'.snapshot_commit').read_text().strip()==proof['observational_commit'] and
            (base/'.snapshot_commit').read_text().strip()==proof['base_commit'], 'observational runtime identity changed')
    require(all(r['validation_key']['code_sha256']==file_digest(base/'.snapshot_commit') for r in records),
            'observational runtime does not preserve these canaries native runtime')
    allowed={'ispatialgym/sampling/campaign_runtime.py','ispatialgym/multienv/campaign_rendering.py',
             'scripts/campaign_20261110/training_batch_probe.py'}
    declared = set(proof['changed_files'])
    require(declared and declared <= allowed and set(proof['changed_payloads']) == declared,
            'observational runtime changes exceed the approved instrumentation')
    changed=set()
    for folder in ('ispatialgym','scripts'):
        paths={str(p.relative_to(tree)) for tree in (base,runtime) for p in (tree/folder).rglob('*.py')}
        for relative in paths:
            a,b=base/relative,runtime/relative
            if not a.is_file() or not b.is_file() or file_digest(a)!=file_digest(b):changed.add(relative)
    require(changed == declared, 'native implementation changed outside observation-only files')
    for relative in declared:
        require(file_digest(runtime/relative)==proof['changed_payloads'][relative],'observational payload changed')
    def physical_ast(path):
        tree=ast.parse(path.read_text())
        tree.body=[node for node in tree.body if not
            (isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in ('verify_loaded','restoration_observation'))
            and not (isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and
                t.id.startswith('_RESTORATION_PROBE_') for t in node.targets))]
        return ast.dump(tree,include_attributes=False)
    require(physical_ast(base/'ispatialgym/sampling/campaign_runtime.py')==
            physical_ast(runtime/'ispatialgym/sampling/campaign_runtime.py'),
            'physical campaign runtime changed alongside instrumentation')
    return {'path':str(proof_path),'sha256':ref['sha256']}


def inputs(root,request):
    root=Path(root).resolve();value=json.loads(Path(request).read_text())
    require(value['schema']=='isg-training-batch-probe-request-v1' and value['campaign_id']=='campaign_20261110','invalid native probe request')
    records=[]
    for ref in value['canaries']:
        path=Path(ref['path']).resolve()
        require(path.is_relative_to(root) and file_digest(path)==ref['sha256'],'canary review changed')
        review=json.loads(path.read_text());require(review['validation_passed'] and not review['accepted'],'input not a validation-only canary')
        row=review['export_record']
        for artifact in row['artifacts'].values():
            path=Path(artifact['path']).resolve()
            require(path.is_relative_to(root) and file_digest(path)==artifact['sha256'],'probe input changed')
        records.append(row)
    require(len(records)==2 and records[0]['instance_id']!=records[1]['instance_id'],'probe needs two distinct canaries')
    if value.get('selected_pool_world') is True:
        from .training_pool_probe import verify_inputs
        verify_inputs(root, value, records)
    elif value.get('sequential_augmentation') is True:
        from ispatialgym.oracle.campaign_batch import record_scope
        record_scope(records)
    else:
        require(records[0]['render_compatibility_sha256']==records[1]['render_compatibility_sha256'],
                'probe cannot bypass production configuration isolation')
    return value,records


def shared_world_proof(runner, records, *, sequential_augmentation=False):
    """Use the production signature and saved-scene checks for template copies."""
    if sequential_augmentation:
        from ispatialgym.oracle.campaign_batch import saved_world_scope
        contract=saved_world_scope(runner,records)
        return {'signature':contract['schema'], 'compatible_template_dirs':contract['compatible_template_dirs'],
                'sequential_augmentation':contract, 'method':contract['method']}
    paths=[r['artifacts']['tro']['path'] for r in records]
    signatures=[runner.instance_signature(path) for path in paths]
    require(signatures[0]==signatures[1],'production world signatures differ')
    directories=[runner.resolve_instance(path)[1] for path in paths]
    compatible,reason=runner.template_compatible(directories[0],paths[0],directories[1],paths[1])
    require(compatible,'production saved scenes differ: '+reason)
    return {'signature':signatures[0],'compatible_template_dirs':directories,
            'scene_check':reason,'method':'native instance_signature + template_compatible'}


def observe_reset_boundary(scene, sim, load, observe, completed=None):
    """Read immediately before the existing reset step, without changing it.

    This diagnostic checkpoint does not replace the post-step restoration
    checks. Both the native call and its arguments are preserved exactly.
    """
    observations=[]
    reset=scene.reset
    raw_reset=vars(scene).get('reset')
    had_reset='reset' in vars(scene)
    def observed_reset(*args, **kwargs):
        step=sim.step_physics
        raw_step=vars(sim).get('step_physics')
        had_step='step_physics' in vars(sim)
        def observed_step(*step_args, **step_kwargs):
            observations.append(observe())
            result=step(*step_args, **step_kwargs)
            if completed is not None:completed.append(observe())
            return result
        sim.step_physics=observed_step
        try:
            return reset(*args, **kwargs)
        finally:
            if had_step: sim.step_physics=raw_step
            else: del sim.step_physics
    scene.reset=observed_reset
    try:
        result=load()
    finally:
        if had_reset: scene.reset=raw_reset
        else: del scene.reset
    require(len(observations)==1, 'diagnostic expected exactly one native reset physics step')
    return result, observations[0]


def observe_next_native_step(sim, observe, destination):
    """Observe one already-requested native step; never add or replace physics."""
    step=sim.step_physics;raw_step=vars(sim).get('step_physics');had_step='step_physics' in vars(sim)
    def observed(*args, **kwargs):
        before=observe()
        try:
            result=step(*args, **kwargs)
            after=observe()
            destination.update(phase='first_native_physics_step_after_completed_loader',
                before=before,after=after,physics_steps_added=0,diagnostic_only=True)
            return result
        finally:
            if had_step:sim.step_physics=raw_step
            else:del sim.step_physics
    sim.step_physics=observed


def observe_next_native_control_step(env, observe, destination):
    """Observe the first existing env.step, which may contain several physics steps.

    Remove the one-shot wrapper before calling native code. The returned cleanup
    also removes an unused wrapper if an episode exits before its first action.
    """
    step=env.step;raw_step=vars(env).get('step');had_step='step' in vars(env)
    def cleanup():
        if vars(env).get('step') is observed:
            if had_step:env.step=raw_step
            else:del env.step
    def observed(*args, **kwargs):
        cleanup()
        before=observe()
        result=step(*args, **kwargs)
        after=observe()
        destination.update(phase='first_native_control_step_after_completed_loader',
            method='env.step',before=before,after=after,physics_steps_added=0,
            native_control_calls_observed=1,diagnostic_only=True)
        return result
    env.step=observed
    return cleanup


class ControlledEpisodeAbort(RuntimeError):
    """A named diagnostic interruption handled by the ordinary episode finally."""


class ControlledEpisodeFailure(RuntimeError):
    """Exercise an ordinary failed episode without changing any requested step."""


class ControlledEpisodeInterrupt(KeyboardInterrupt):
    """Exercise the native BaseException cleanup path in the bounded test only."""


def abort_after_native_control_step(env, destination, exception_type=ControlledEpisodeAbort):
    """Complete one unchanged requested action, then exercise native error cleanup."""
    step = env.step
    original = vars(env).get('step')
    had_step = 'step' in vars(env)
    def cleanup():
        if vars(env).get('step') is interrupted:
            if had_step: env.step = original
            else: del env.step
    def interrupted(*args, **kwargs):
        cleanup()
        step(*args, **kwargs)
        destination.update(native_control_calls_completed=1, physics_steps_added=0,
                           exception_type=exception_type.__name__, injected=True)
        raise exception_type('bounded restoration cleanup control after one native action')
    env.step = interrupted
    return cleanup


def install_reset_boundary_probe(runner, *, final_pre_action=False, next_native_step=False,
                                 next_native_control_step=False, pending_cleanups=None, observer=None,
                                 abort_control=None):
    """Instrument only this diagnostic process; the ordinary loader is unchanged."""
    restore=runner.InstanceLoader.restore
    def observed_restore(loader, *args, **kwargs):
        if observer is None:
            from ispatialgym.sampling.campaign_runtime import restoration_observation
        else:
            restoration_observation=observer
        task=loader.task
        load=task.load_tro_instance
        raw_load=vars(task).get('load_tro_instance')
        had_load='load_tro_instance' in vars(task)
        checkpoints=[];completed=[]
        def observed_load(*load_args, **load_kwargs):
            result, checkpoint=observe_reset_boundary(loader.env.scene, runner.og.sim,
                lambda:load(*load_args, **load_kwargs),
                lambda:restoration_observation(loader.tro.meta.raw['campaign'], loader.env.scene, runner.og.sim),
                completed=completed)
            checkpoints.append(checkpoint)
            return result
        task.load_tro_instance=observed_load
        try:
            result=restore(loader, *args, **kwargs)
        finally:
            if had_load: task.load_tro_instance=raw_load
            else: del task.load_tro_instance
        require(len(checkpoints)==1, 'diagnostic did not observe exactly one instance load')
        loader.report['campaign_restoration']['native_reset_boundary_observation']={
            'phase':'after_scene_restore_before_existing_reset_physics_step',
            'diagnostic_only':True, 'post_step_checks_unchanged':True,
            'observation':checkpoints[0]}
        require(len(completed)==1, 'diagnostic did not observe the completed native reset step')
        loader.report['campaign_restoration']['native_reset_step_completed_observation']={
            'phase':'immediately_after_existing_scene_reset_physics_step',
            'diagnostic_only':True,'physics_steps_added':0,'observation':completed[0]}
        if final_pre_action:
            # The entire native loader has returned, including constraints,
            # grasps, settling, final reset, and its pose checks. Read only:
            # retain the earlier checkpoint and add no simulator step.
            loader.report['campaign_restoration']['native_final_pre_action_observation']={
                'phase':'after_complete_native_loader_before_task_actions',
                'diagnostic_only':True, 'earlier_checks_unchanged':True,
                'physics_steps_added':0,
                'observation':restoration_observation(loader.tro.meta.raw['campaign'],
                                                       loader.env.scene, runner.og.sim)}
        if next_native_step:
            destination={}
            loader.report['campaign_restoration']['next_native_physics_step_observation']=destination
            observe_next_native_step(runner.og.sim,
                lambda:restoration_observation(loader.tro.meta.raw['campaign'],loader.env.scene,runner.og.sim),destination)
        if next_native_control_step:
            destination={}
            loader.report['campaign_restoration']['next_native_control_step_observation']=destination
            cleanup=observe_next_native_control_step(loader.env,
                lambda:restoration_observation(loader.tro.meta.raw['campaign'],loader.env.scene,runner.og.sim),destination)
            if pending_cleanups is not None:pending_cleanups.append(cleanup)
        if abort_control is not None and abort_control.get('enabled') is True:
            destination = {}
            loader.report['campaign_restoration']['controlled_abort'] = destination
            cleanup = abort_after_native_control_step(loader.env, destination,
                abort_control.get('exception_type', ControlledEpisodeAbort))
            if pending_cleanups is not None: pending_cleanups.append(cleanup)
        return result
    runner.InstanceLoader.restore=observed_restore
    return restore


def episode_order(value, kind):
    """A repeated-A control is diagnostic only and cannot pass the A/B/A gate."""
    if value.get('sequential_render_only'):
        require(kind=='kinematic_observation_rendering' and value.get('sequential_augmentation') is True and
                value.get('render_cleanup_only') is not True and value.get('production_released') is False and
                value.get('diagnostic_control') in (None,'FRESH_A','FRESH_B'),
                'sequential render scope cannot authorize physics or another control')
    control = value.get('diagnostic_control')
    if control is None:
        return [0, 1, 0]
    if control in ('RENDER_FAILURE_A','RENDER_INTERRUPT_A'):
        require(kind=='kinematic_observation_rendering' and value.get('render_cleanup_only') is True and
                value.get('sequential_augmentation') is True and value.get('diagnostic_only') is True and
                value.get('production_released') is False,'render cleanup control is not an explicit diagnostic')
        return [0,0]
    allowed_physics=('A_A', 'A_ABORT_A', 'A_FAILURE_A', 'A_INTERRUPT_A', 'FRESH_A', 'FRESH_B')
    require(control in allowed_physics and
            (kind == 'physics_action_open_loop' or
             (control in ('FRESH_A', 'FRESH_B') and value.get('sequential_augmentation') is True))
            and value.get('diagnostic_only') is True
            and value.get('production_released') is False,
            'unsupported or non-diagnostic restoration control')
    return [0] if control == 'FRESH_A' else ([1] if control == 'FRESH_B' else [0, 0])


@contextmanager
def world_construction_counter(og):
    """Count actual native Environment calls; preserve every argument/result."""
    constructor=og.Environment
    proof={'attempted':0,'completed':0,'method':'omnigibson.Environment', 'instrumentation_only':True}
    def observed(*args,**kwargs):
        proof['attempted']+=1
        result=constructor(*args,**kwargs)
        proof['completed']+=1
        return result
    og.Environment=observed
    try:yield proof
    finally:og.Environment=constructor


def run(root,request,output,kind):
    import omnigibson as og
    with world_construction_counter(og) as construction:
        return _run(root,request,output,kind,construction)


def _run(root,request,output,kind,construction):
    root=Path(root).resolve();output=Path(output).resolve()
    require(output.is_relative_to(root) and not output.exists(),'probe output must be new and private')
    value,records=inputs(root,request);output.mkdir(parents=True)
    require(not (value.get('render_cleanup_only') or value.get('sequential_render_only')) or kind=='kinematic_observation_rendering',
            'render cleanup helper cannot execute physics')
    require(not value.get('sequential_checker_only') or kind=='physics_action_open_loop',
            'physics saved-scene checker cannot execute rendering')
    require(not value.get('sequential_anchor_rebase') or kind=='physics_action_open_loop',
            'prospective anchor repair cannot execute rendering')
    runtime=Path(__file__).resolve().parents[2]/'.snapshot_commit'
    require(file_digest(runtime)==value['probe_runtime_sha256'],'probe runtime differs from approved observational snapshot')
    compatibility=verify_runtime_compatibility(root,value,records,runtime.parent)
    # Exact production-purpose sources retain their original native identity;
    # the loader independently rechecks this narrow observational proof.
    os.environ['ISPATIALGYM_TRAINING_OBSERVATION_REQUEST']=str(Path(request).resolve())
    os.environ['ISPATIALGYM_TRAINING_OBSERVATION_REQUEST_SHA256']=file_digest(request)
    require(not value.get('physical_candidate') or kind=='physics_action_open_loop',
            'pristine physical candidate is physics-only')
    os.environ['ISPATIALGYM_CAMPAIGN_RESTORATION_PROBE']='1'
    if value.get('observe_native_dynamics') is True:
        os.environ['ISPATIALGYM_CAMPAIGN_RESTORATION_CONTEXT']='1'
    order=episode_order(value,kind);entries=[]
    common={'schema':'isg-training-restoration-probe-v1','campaign_id':'campaign_20261110','kind':kind,
            'request':{'path':str(Path(request).resolve()),'sha256':file_digest(request)},
            'probe_runtime':{'path':str(runtime),'sha256':file_digest(runtime)},'entries':entries,
            'runtime_compatibility':compatibility,
            'world_construction':construction,
            'accepted':False,'eligible_for_goal':False,'quota_consumed':0}
    if value.get('diagnostic_control'):
        common.update(schema='isg-training-restoration-control-v1', diagnostic_control=value['diagnostic_control'],
                      release_eligible=False, scope='Repeated-A baseline only; cannot satisfy the A/B/A restoration gate.')
    if kind=='physics_action_open_loop':
        # A and B are immutable copies of one exact configuration. Production
        # prebuild checks compare their content and recorded scene, not paths.
        os.environ['ISPATIALGYM_FAST_COLLECTOR_PREBUILD']='1'
        from ispatialgym.oracle import run_isg
        if value.get('explicit_physics_sdk_rebuild_v1') is True:
            from .training_physics_rebuild_probe import install as install_sdk_rebuild, DELTA as SDK_DELTA
            install_sdk_rebuild(run_isg, value, output)
            common['physics_sdk_reconstruction_delta'] = SDK_DELTA
        if value.get('fixed_mass_baseline') is True:
            from .training_fixed_mass_probe import install_fixed_mass_baseline, DELTA
            install_fixed_mass_baseline(run_isg,value,output)
            common['physical_delta']=DELTA
            common['fixed_mass_audit']=str(output/'fixed_mass_audit.json')
        if value.get('reset_contact_cache_before_step_v1') is True:
            from .training_contact_reset_probe import install as install_contact_reset, DELTA as CONTACT_DELTA
            install_contact_reset(run_isg, value, output)
            common['contact_reset_delta'] = CONTACT_DELTA
        if value.get('physical_candidate'):
            from .training_pristine_probe import install_override
            install_override(run_isg, value)
            common.update(schema='isg-training-physical-diagnostic-v1', release_eligible=False,
                          physical_candidate=value['physical_candidate'])
        from ispatialgym.fastflags import fast_on
        require(fast_on('COLLECTOR_PREBUILD'),'production template compatibility is disabled')
        pending_cleanups=[]
        abort_control = {'enabled': False}
        interruption_types={'A_ABORT_A':ControlledEpisodeAbort,'A_FAILURE_A':ControlledEpisodeFailure,
                            'A_INTERRUPT_A':ControlledEpisodeInterrupt}
        if value.get('diagnostic_control') in interruption_types:
            require(value.get('observe_reset_boundary') is True and
                    value.get('observe_final_pre_action') is True and
                    value.get('observe_next_native_control_step') is True,
                    'controlled abort requires all native boundary observations')
        if value.get('observe_reset_boundary') is True:
            install_reset_boundary_probe(run_isg,
                final_pre_action=value.get('observe_final_pre_action') is True,
                next_native_step=value.get('observe_next_native_physics_step') is True,
                next_native_control_step=value.get('observe_next_native_control_step') is True,
                pending_cleanups=pending_cleanups, abort_control=abort_control)
            common['reset_boundary_diagnostic']=True
            common['final_pre_action_diagnostic']=value.get('observe_final_pre_action') is True
        world=shared_world_proof(run_isg,records,sequential_augmentation=value.get('sequential_augmentation') is True)
        common['world_compatibility']=world
        shared={'signature':world['signature'],
                'compatible_template_dirs':set(map(os.path.abspath,world['compatible_template_dirs']))}
        if world.get('sequential_augmentation'):
            shared['sequential_augmentation']=world['sequential_augmentation']
        for ordinal,index in enumerate(order):
            abort_control['enabled'] = value.get('diagnostic_control') in interruption_types and ordinal == 0
            abort_control['exception_type']=interruption_types.get(value.get('diagnostic_control'),ControlledEpisodeAbort)
            row=records[index];destination=output/('episode_'+str(ordinal))
            argv=['--instance',row['artifacts']['tro']['path'],'--mode','replay',
                  '--replay',row['artifacts']['demo']['path'],'--record','hdf5','--no-goal-override','--out',str(destination)]
            args=run_isg.parse_args(argv)
            try:
                code,result=run_isg.run_instance(args,argv[1],shared)
            finally:
                while pending_cleanups:pending_cleanups.pop()()
            report=destination/row['instance_id']/'meta.json'
            entries.append({'canary_review':value['canaries'][index],'returncode':code,'result':result,
                            'report':{'path':str(report),'sha256':file_digest(report)}})
            write_json_atomic(output/'probe.json',common)
            if abort_control['enabled']:
                saved = json.loads(report.read_text())
                interruption = saved.get('loader', {}).get('campaign_restoration', {}).get('controlled_abort', {})
                require(code != 0 and result.get('success') is False and
                        result.get('error', '').startswith(abort_control['exception_type'].__name__+'(') and
                        interruption.get('injected') is True and
                        interruption.get('exception_type')==abort_control['exception_type'].__name__ and
                        interruption.get('native_control_calls_completed') == 1,
                        'native episode did not execute the exact controlled-abort cleanup path')
                entries[-1]['expected_controlled_abort'] = True
                write_json_atomic(output/'probe.json', common)
            else:
                require(code==0 and result.get('success') is True,'native batch physics episode failed')
    else:
        require(kind=='kinematic_observation_rendering','unknown native probe kind')
        from ispatialgym.multienv import replay
        manifest=output/'canaries.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n'for r in records))
        listing=output/'demos.txt';listing.write_text(''.join(str(Path(records[i]['artifacts']['demo']['path']).parent)+'\n'for i in order))
        results=output/'results.jsonl'
        extra=['--sequential-augmentation'] if value.get('sequential_augmentation') is True else []
        if value.get('render_cleanup_only'):
            require(value.get('render_fault_boundary','writer') in ('writer','post_render'),'unsupported render fault boundary')
            extra+=['--diagnostic-episode-fault', 'failure' if value['diagnostic_control']=='RENDER_FAILURE_A' else 'interruption',
                    '--diagnostic-fault-boundary',value.get('render_fault_boundary','writer')]
        code=replay.main(['--demos','@'+str(listing),'--campaign-manifest',str(manifest),
            '--root',str(output/'rendered'),'--task',records[0]['variant'],'--slots','1',
            '--scene','recorded','--results',str(results),'--repo-prefix','training_probe']+extra)
        rows=[(i,json.loads(line))for i,line in enumerate(results.read_text().splitlines())]
        episodes=[(i,row)for i,row in rows if row.get('tag') in ('REPLAY_DEMO','REPLAY_EPISODE_ABORTED')]
        outputs=[{'path':str(p),'sha256':file_digest(p)}for p in sorted((output/'rendered').rglob('*'))if p.is_file()]
        require(code==0 and len(episodes)==len(order) and outputs,'native renderer did not produce the requested episodes and files')
        for ordinal,(line,row)in enumerate(episodes):
            require(Path(row['demo']).name==records[order[ordinal]]['instance_id'],'render order differs')
            failure=value.get('render_cleanup_only') is True and ordinal==0
            require((row.get('tag')=='REPLAY_EPISODE_ABORTED' and row['cleanup']['expected_controlled_failure'] is True)
                    if failure else row.get('tag')=='REPLAY_DEMO','unexpected renderer failure outcome')
            entries.append({'canary_review':value['canaries'][order[ordinal]],
                'report':{'path':str(results),'sha256':file_digest(results),'jsonl_index':line},'outputs':outputs,
                **({'expected_controlled_failure':True}if failure else {})})
        write_json_atomic(output/'probe.json',common)
    # Recheck all source bytes after running; no source state or goal rewrites.
    inputs(root,request)
    require(construction['attempted']==construction['completed']==1,
            'batch did not execute exactly one native Environment construction')
    return common


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','request','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--kind',required=True,choices=('physics_action_open_loop','kinematic_observation_rendering'))
    a=p.parse_args()
    try:print(json.dumps(run(a.root,a.request,a.output,a.kind)),flush=True)
    except BaseException as error:
        # OG shutdown may terminate the interpreter before Python prints a
        # traceback. Preserve the actual diagnostic failure first.
        import traceback
        destination=Path(a.output)
        if destination.is_dir():
            write_json_atomic(destination/'failure.json',{'schema':'isg-training-batch-probe-failure-v1',
                'kind':a.kind,'error':repr(error),'traceback':traceback.format_exc(),
                'accepted':False,'eligible_for_goal':False,'quota_consumed':0})
        raise
    finally:
        import sys
        og=sys.modules.get('omnigibson')
        if og is not None and hasattr(og,'shutdown'):og.shutdown()
