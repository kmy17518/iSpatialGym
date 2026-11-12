"""Scoped production evidence and exact capacity binding; no release by a flag.

This verifier never creates a release or a ticket. Shared implementation proof,
applicable task-path proof, and exact configuration proof are separate inputs.
Unfinished unrelated configurations do not participate in this decision.
"""
from pathlib import Path
import json

from .contracts import digest, file_digest, require
from .training_restoration import checked, validated_canary, verify_batch, same_loaded_state


def _reference(root, value):
    path = Path(value['path']).resolve()
    require(path.is_relative_to(Path(root).resolve()) and path.is_file()
            and file_digest(path) == value['sha256'], 'release artifact missing or changed')
    return path


def verify_probe_execution(root, reference, execution_reference):
    """A complete probe followed by a teardown crash is not a clean path run.

    Keep the physical comparator independent. Production release additionally
    binds the native observations to the submitted output and raw scheduler row.
    """
    probe = checked(root, reference)
    execution = checked(root, execution_reference)
    require(execution.get('schema') == 'isg-batch-execution-observation-v1' and
            execution.get('request') == probe['request'], 'probe execution request differs')
    request = checked(root, probe['request'])
    require(request.get('schema') == 'isg-training-batch-probe-request-v1', 'unknown probe execution request')
    kind = probe['kind']; job_ids = set()
    for entry in probe['entries']:
        report = checked(root, entry['report'])
        restoration = (report['loader']['campaign_restoration'] if kind == 'physics_action_open_loop'
                       else report['campaign_restoration'])
        observation = (restoration['native_final_pre_action_observation']['observation']
                       if kind == 'physics_action_open_loop' else restoration['native_observation'])
        job_ids.add(str(observation['scheduler_job_id']))
    require(len(job_ids) == 1 and next(iter(job_ids)).isdigit(), 'probe has no unique native allocation')
    job_id = next(iter(job_ids))
    jobs = [row for row in execution['jobs'] if str(row.get('job_id')) == job_id and row.get('kind') == kind]
    require(len(jobs) == 1 and jobs[0].get('state') == 'COMPLETED' and jobs[0].get('exit_code') == '0:0',
            'native probe process did not complete cleanly')
    submission = checked(root, execution['submission'])
    require(submission.get('request') == probe['request'], 'probe submission request differs')
    submitted = [row for row in submission['jobs'] if str(row.get('job_id')) == job_id and row.get('kind') == kind]
    require(len(submitted) == 1 and submitted[0].get('returncode') == 0 and
            Path(submitted[0]['output']).resolve() == Path(reference['path']).resolve().parent,
            'probe execution output or submitted allocation differs')
    scheduler = execution['scheduler_evidence']; raw = _reference(root, scheduler).read_text()
    command = scheduler.get('command', [])
    required_fields = {'JobIDRaw', 'State', 'ExitCode', 'Start', 'End', 'ElapsedRaw', 'AllocTRES'}
    formats = [arg.split('=', 1)[1].split(',') for arg in command if arg.startswith('--format=')]
    require(command and Path(command[0]).name == 'sacct' and '-X' in command and '-P' in command and
            '-n' in command and len(formats) == 1 and len(formats[0]) == len(set(formats[0])) and
            required_fields <= set(formats[0]) <= required_fields | {'JobID'} and '-j' in command and
            job_id in command[command.index('-j')+1].split(','), 'exact raw scheduler query required')
    fields = formats[0]
    rows = [line.split('|') for line in raw.splitlines() if line.strip()]
    require(all(len(row) == len(fields) for row in rows), 'raw scheduler columns differ from exact query')
    matches = [dict(zip(fields, row)) for row in rows if row[fields.index('JobIDRaw')] == job_id]
    require(len(matches) == 1, 'native allocation raw scheduler row missing or ambiguous')
    row = matches[0]
    require(row.get('JobID', job_id) == job_id and row['State'] == 'COMPLETED' and row['ExitCode'] == '0:0' and
            row['Start'] not in ('', 'Unknown', 'None') and row['End'] not in ('', 'Unknown', 'None') and
            row['ElapsedRaw'].isdigit() and int(row['ElapsedRaw']) > 0 and row['AllocTRES'],
            'native raw scheduler result is not a clean completed execution')
    require(jobs[0].get('start') == row['Start'] and jobs[0].get('end') == row['End'] and
            jobs[0].get('elapsed_seconds') == int(row['ElapsedRaw']) and jobs[0].get('allocated_resources') == row['AllocTRES'],
            'probe execution summary differs from raw scheduler evidence')
    return {'probe': reference, 'execution': execution_reference, 'job_id': job_id,
            'status': 'native_process_completed_cleanly'}


def verify_render(root, reference, canary):
    """Reconcile actual native rendering again, including exact exported inputs."""
    from .training_pipeline_status import template_render_observations
    request = checked(root, reference)
    require(request.get('schema') == 'isg-training-template-render-request-v1', 'exact render request required')
    matches = [r for r in request['records'] if r.get('instance_id') == canary['instance_id']]
    require(len(matches) == 1, 'exact canary render missing or duplicated')
    row = matches[0]
    require(row['configuration_sha256'] == canary['source_proof']['config']['sha256'],
            'render configuration differs')
    historical = validated_canary(root, row['canary_review'], 1794321600)
    require(digest(historical['export_record']) == digest(canary['export_record']),
            'render belongs to another exact canary')
    for key in ('configuration', 'source_environment', 'native_runtime_identity', 'manifest', 'demos'):
        _reference(root, row[key])
    require(row['native_runtime_identity']['sha256'] == canary['export_record']['validation_key']['code_sha256'],
            'render native runtime differs')
    directory = Path(row['results']).resolve().parent
    require(directory.is_relative_to(Path(reference['path']).resolve().parent), 'render output outside request')
    observed = template_render_observations(root, reference['path'])
    rendered = observed[(row['candidate_id'], int(row['start']))]
    require(rendered['status'] == 'verified', 'exact template native rendering failed')
    execution = json.loads((directory/'execution_observation.json').read_text())
    require(execution.get('returncode') == execution.get('reconciliation_returncode') == 0 and
            execution.get('candidate_id') == row['candidate_id'] and execution.get('job_id') == rendered['job_id']
            and execution.get('finished_at', 0) > execution.get('started_at', 0), 'native rendering completion unproved')
    native_rows = [json.loads(line) for line in Path(row['results']).read_text().splitlines()]
    actual = checked(root, rendered['result'])
    require([x for x in native_rows if x.get('tag') == 'REPLAY_DEMO'] == actual['episodes'] and
            any(x.get('tag') == 'REPLAY_SUMMARY' for x in native_rows), 'native rendering result differs from raw episodes')
    return {'request': reference, 'instance_id': canary['instance_id']}


def _observation(report, name):
    restoration = report['loader']['campaign_restoration']
    value = restoration[name]['observation']
    require(value['loaded_state_sha256'] == digest(value['loaded_state']), 'native boundary state changed')
    require(value.get('scheduler_job_id') and value.get('process_session') and
            value.get('simulator_module', '').startswith('omnigibson.'), 'boundary is not scheduled native evidence')
    return value


def same_reset_boundary(first, last):
    """Keep the existing physical tolerances and compare actual reset masses."""
    # Lighting is installed later by the unchanged augmentation loader. Early
    # checkpoints compare physics; the completed pre-action boundary below
    # still checks the full lights/skybox state through same_loaded_state.
    physical = lambda observation: {'objects':observation['loaded_state']['objects'],
        'lights':{}, 'skybox':{'intensity':0., 'color':[0.,0.,0.]}}
    same_loaded_state(physical(first), physical(last))
    for observation in (first, last):
        context = observation.get('native_dynamics_context')
        require(isinstance(context, dict) and context.get('physics_steps_added') == 0 and
                observation.get('native_dynamics_context_sha256') == digest(context),
                'reset boundary native dynamics context missing or changed')
    a = first['native_dynamics_context']['dynamic_roots']
    b = last['native_dynamics_context']['dynamic_roots']
    require(a and a.keys() == b.keys(), 'reset native actor inventory differs')
    for name in a:
        for field in ('mass', 'sleep_threshold', 'solver_position_iteration_count',
                      'solver_velocity_iteration_count'):
            require(type(a[name].get(field)) in (int, float) and a[name][field] == b[name].get(field),
                    'reset native '+field+' differs')


def verify_cold_reset_compatibility(root, reference, started_at, native, task):
    """Matched old/new cold inputs prove a fix preserves the original first reset.

    This evidence is additional to exact-config controls. The physical diagnostic
    is never substituted for their native/demo/replay or configuration identity.
    """
    from .acceptance import report_passed
    from .training_batch_probe import inputs, verify_runtime_compatibility
    value = checked(root, reference)
    require(value.get('schema') == 'isg-training-cold-reset-compatibility-v1',
            'original cold-reset comparison required')
    cases = []
    for name in ('original', 'candidate'):
        probe = checked(root, value[name]); request_ref = probe['request']
        _reference(root, request_ref)
        request, records = inputs(root, request_ref['path'])
        require(probe['kind'] == 'physics_action_open_loop' and probe['entries'],
                'cold comparison requires executed native physics')
        identity = probe['probe_runtime']; runtime = _reference(root, identity).parent
        require(identity['sha256'] == request['probe_runtime_sha256'] and
                verify_runtime_compatibility(root, request, records, runtime) == probe['runtime_compatibility'],
                'cold comparison runtime is unproved')
        if name == 'original':
            require(not request.get('physical_candidate'), 'cold baseline already changes native physics')
        else:
            require(request.get('physical_candidate') and request.get('diagnostic_control') == 'FRESH_A'
                    and len(probe['entries']) == 1, 'candidate comparison must be a fresh matched physical diagnostic')
            for relative in ('ispatialgym/oracle/run_isg.py', 'ispatialgym/oracle/campaign_restore.py'):
                require(file_digest(runtime/relative) == file_digest(Path(native)/relative),
                        'cold comparison did not execute the production restoration implementation')
        entry = probe['entries'][0]
        require(entry['canary_review'] == request['canaries'][0] and entry['returncode'] == 0,
                'cold comparison first episode failed or selected another source')
        source = validated_canary(root, entry['canary_review'], started_at)
        require(source['task'] == task, 'cold compatibility belongs to another task path')
        report = checked(root, entry['report']); report_passed(report, 'replay')
        require(report['instance_id'] == source['instance_id'] and
                Path(report['replay']).resolve() == Path(source['export_record']['artifacts']['demo']['path']).resolve(),
                'cold comparison did not execute the unchanged source actions')
        boundaries = [_observation(report, key) for key in (
            'native_reset_boundary_observation', 'native_reset_step_completed_observation',
            'native_final_pre_action_observation')]
        if name == 'candidate':
            restored = report['loader'].get('pre_reset_pristine_mass', {})
            require(restored.get('revision') == 'pristine_mass_before_reset_v1' and
                    restored.get('captured_at') == 'build_env_before_first_restore' and
                    restored.get('actors_checked', 0) > 0 and restored.get('actors_restored') == [] and
                    restored.get('physics_steps_added') == 0 and
                    restored.get('later_serialized_mass_application_unchanged') is True,
                    'fresh candidate changed original cold mass/inertia or loader order')
        step = report['loader']['campaign_restoration']['next_native_control_step_observation']
        require(step['native_control_calls_observed'] == 1 and step['physics_steps_added'] == 0,
                'cold comparison first native action is unproved')
        for endpoint in ('before', 'after'):
            require(step[endpoint]['loaded_state_sha256'] == digest(step[endpoint]['loaded_state']),
                    'cold comparison action state changed')
        same_loaded_state(boundaries[-1]['loaded_state'], step['before']['loaded_state'])
        cases.append((source['export_record'], boundaries, step))
    old, new = cases
    require(digest(old[0]) == digest(new[0]), 'cold comparison source TRO/template/actions changed')
    require(old[1][0]['process_session'] != new[1][0]['process_session'], 'cold comparison reused a prior process')
    for first, last in zip(old[1], new[1]):
        same_reset_boundary(first, last)
    same_loaded_state(old[1][-1]['loaded_state'], new[1][-1]['loaded_state'])
    same_reset_boundary(old[2]['after'], new[2]['after'])
    return {'reference': reference, 'status': 'matched_original_cold_reset_verified',
            'source_instance': old[0]['instance_id'], 'production_authorization': False}


def verify_controls(root, references, started_at):
    """Fresh A, successful AA, aborted AA and ABA must agree at action boundaries."""
    from .acceptance import report_passed
    from .training_batch_probe import inputs, verify_runtime_compatibility
    require(set(references) == {'fresh_A', 'successful_AA', 'aborted_AA', 'ABA'},
            'all four discriminating restoration controls required')
    expected = {'fresh_A': ('FRESH_A', [0]), 'successful_AA': ('A_A', [0, 0]),
                'aborted_AA': ('A_ABORT_A', [0, 0]), 'ABA': (None, [0, 1, 0])}
    results = {}; baselines = []; exact_A = None; runtime = None
    for kind, reference in references.items():
        probe = checked(root, reference); request_ref = probe['request']
        _reference(root, request_ref)
        request, records = inputs(root, request_ref['path'])
        require(not request.get('physical_candidate'), 'physical diagnostic override is not exact-config release evidence')
        control, order = expected[kind]
        require(request.get('diagnostic_control') == control and
                probe['kind'] == 'physics_action_open_loop' and len(probe['entries']) == len(order),
                'restoration control sequence differs')
        identity = probe['probe_runtime']; native = _reference(root, identity)
        require(identity['sha256'] == request['probe_runtime_sha256'], 'control runtime differs')
        compatibility = verify_runtime_compatibility(root, request, records, native.parent)
        require(probe['runtime_compatibility'] == compatibility, 'control compatibility missing')
        require(runtime is None or runtime == identity, 'controls used different physical runtimes')
        runtime = identity
        reports = []; observations = []
        for ordinal, (entry, member) in enumerate(zip(probe['entries'], order)):
            require(entry['canary_review'] == request['canaries'][member], 'control selected wrong source')
            source = validated_canary(root, entry['canary_review'], started_at)
            if member == 0:
                exact_A = exact_A or source['export_record']
                require(digest(source['export_record']) == digest(exact_A), 'fresh/reused A inputs differ')
            report = checked(root, entry['report'])
            require(report['instance_id'] == source['instance_id'] and
                    Path(report['replay']).resolve() == Path(source['export_record']['artifacts']['demo']['path']).resolve(),
                    'control did not replay the exact source actions')
            if kind == 'aborted_AA' and ordinal == 0:
                abort = report['loader']['campaign_restoration']['controlled_abort']
                require(entry.get('expected_controlled_abort') is True and entry['returncode'] != 0
                        and report.get('error', '').startswith('ControlledEpisodeAbort(')
                        and abort == {'native_control_calls_completed': 1, 'physics_steps_added': 0,
                                      'exception_type': 'ControlledEpisodeAbort', 'injected': True},
                        'failed-episode cleanup was not the exact controlled interruption')
            else:
                report_passed(report, 'replay')
                require(entry['returncode'] == 0, 'native control process failed')
            boundaries = [_observation(report, key) for key in (
                'native_reset_boundary_observation', 'native_reset_step_completed_observation',
                'native_final_pre_action_observation')]
            final = boundaries[-1]
            require(final['validation_key'] == source['export_record']['validation_key'], 'boundary config differs')
            step = report['loader']['campaign_restoration']['next_native_control_step_observation']
            require(step['native_control_calls_observed'] == 1 and step['physics_steps_added'] == 0,
                    'first native action checkpoint missing')
            for endpoint in ('before', 'after'):
                require(step[endpoint]['loaded_state_sha256'] == digest(step[endpoint]['loaded_state']),
                        'native first-action observation changed')
            same_loaded_state(final['loaded_state'], step['before']['loaded_state'])
            reports.append((final, step, boundaries)); observations.append(final)
        require(len({o['process_session'] for o in observations}) == 1 and
                len({o['sim_runtime_id'] for o in observations}) == 1 and
                len({o['scene_runtime_id'] for o in observations}) == 1, 'control did not reuse its native world')
        for index in range(1, len(observations)):
            require(observations[index-1]['sequence'] < observations[index]['sequence'], 'native sequence order differs')
        if kind == 'ABA':
            require(digest(observations[0]['loaded_state']) != digest(observations[1]['loaded_state']),
                    'ABA did not execute a distinct B state')
        first = reports[0]
        for ordinal, member in enumerate(order):
            if member == 0:
                same_loaded_state(first[0]['loaded_state'], reports[ordinal][0]['loaded_state'])
                same_loaded_state(first[1]['after']['loaded_state'], reports[ordinal][1]['after']['loaded_state'])
                for a, b in zip(first[2], reports[ordinal][2]):
                    same_reset_boundary(a, b)
        baselines.append(first)
        results[kind] = {'reference': reference, 'process_session': observations[0]['process_session']}
    require(len({v['process_session'] for v in results.values()}) == 4, 'fresh controls did not use independent processes')
    for item in baselines[1:]:
        same_loaded_state(baselines[0][0]['loaded_state'], item[0]['loaded_state'])
        same_loaded_state(baselines[0][1]['after']['loaded_state'], item[1]['after']['loaded_state'])
        for a, b in zip(baselines[0][2], item[2]):
            same_reset_boundary(a, b)
    return results


def verify_sampler_pool(root, reference, started_at):
    """Actual sampler switching, active-only exports and all three episode paths."""
    from .training_pipeline_status import sequence_stage_references
    evidence = checked(root, reference)
    result = checked(root, evidence['sequence_result']); request = checked(root, result['request'])
    require(result['sequence'] == ['A', 'B', 'A'] and len(result['stages']) == len(request['records']) == 3,
            'actual three-template sampler sequence missing')
    require(len(evidence['canaries']) == len(evidence['renders']) == 3, 'each sampler export needs full episode evidence')
    targets = []; prior_events = []; loaded = None
    for ordinal, stage in enumerate(result['stages']):
        row = request['records'][ordinal]
        require((stage['candidate_id'], stage['start']) == (row['candidate_id'], row['start']), 'sampler stage identity differs')
        refs = sequence_stage_references(root, Path(evidence['sequence_result']['path']), stage)
        gate = checked(root, refs['native_gate']); journal = checked(root, refs['pool_execution'])
        scene = checked(root, refs['template_export']); own = validated_canary(root, evidence['canaries'][ordinal], started_at)
        require(own['source_proof']['config']['sha256'] == row['config_sha256'] and
                int(own['instance_id'].rsplit('__', 1)[-1]) == row['start'], 'sampler export/canary differs')
        require(gate['returncode'] == 0 and gate['tro_count'] == 1 and
                gate['tros'][0]['sha256'] == own['export_record']['artifacts']['tro']['sha256'], 'native pool export differs')
        from ispatialgym.sampling.model_pool import signature, digest as pool_digest
        manifest = checked(root, journal['manifest'])
        expected_loaded = {f"campaign_pool_{pool_digest(signature(**{k:entry[k] for k in ('category','model','dataset','scale')}, fixed_base=entry.get('fixed_base',False), extra_abilities=entry.get('extra_abilities')))[:16]}_{copy}"
                           for entry in manifest['entries'] for copy in range(entry['copies'])}
        require(expected_loaded and all(set(event['loaded']) == expected_loaded and
                set(event['native_observation']['objects']) == expected_loaded for event in journal['events']),
                'executed pool omits or adds selected physical members')
        require(scene['metadata']['ispatialgym']['model_pool']['manifest_sha256'] == journal['manifest']['sha256'],
                'active-only export belongs to another pool manifest')
        events = journal['events']; require(events[:len(prior_events)] == prior_events, 'pool journal changed between stages')
        require(sum(e['event'] == 'preloaded' for e in events) == 1, 'pool rebuilt between templates')
        loaded = loaded or set(events[0]['loaded'])
        current = events[len(prior_events):]; require(current and current[-1]['event'] == 'released', 'pool cleanup did not complete')
        exported = [e for e in current if e['event'] == 'exported']; require(exported, 'sampler did not export active objects')
        active = set()
        for event in exported:
            observation = event['native_observation']; require(observation['pool_physics_rebuilds'] == 1, 'pool physics rebuilt')
            active.update(event['active']); require(len(active) == 1, 'unexpected active target inventory')
            for name, obj in observation['objects'].items():
                if name in active:
                    require(obj['visible'] and obj['planner_included'], 'active model missing from native task')
                else:
                    require(not obj['visible'] and not obj['planner_included'] and not any(obj['collision_enabled']),
                            'inactive pooled model contaminates native task')
        require(set(scene['objects_info']['init_info']) & loaded == active, 'inactive model leaked into saved scene')
        require(set(scene['metadata']['ispatialgym']['model_pool']['active_names']) == active, 'export active identity differs')
        target = next(o for o in own['source_proof']['template']['objects'] if o['role'] == 'target')
        targets.append((target['category'], target['model'], tuple(target['default_scale'])))
        verify_render(root, evidence['renders'][ordinal], own)
        prior_events = events
    require(targets[0] == targets[2] and targets[0] != targets[1], 'real model sequence was not A/B/A')
    return {'reference': reference, 'targets': targets}


# A witness is useful only for the stage it actually executed. In particular a
# pool canary cannot attest to an unrelated restoration-control implementation.
STAGE_COMPONENTS = {
    'restoration': {'ispatialgym/oracle/run_isg.py', 'ispatialgym/oracle/campaign_restore.py',
                    'ispatialgym/sampling/campaign_runtime.py'},
    'sampler_pool': {'ispatialgym/sampling/sample_isg.py', 'ispatialgym/sampling/model_pool.py',
                     'ispatialgym/sampling/campaign_runtime.py'},
    'rendering': {'ispatialgym/multienv/replay.py', 'ispatialgym/multienv/campaign_rendering.py',
                  'ispatialgym/sampling/campaign_runtime.py'},
}


def restoration_options(root, canary):
    config = checked(root, canary['source_proof']['config'])
    options = config['template'].get('scene_setup', {}).get('restoration_options', {})
    tro = checked(root, canary['export_record']['artifacts']['tro'])['ispatialgym']['campaign']
    require(tro.get('scene_setup', {}).get('restoration_options', {}) == options,
            'executed TRO restoration options differ from exact configuration')
    require(all(type(v) is bool for v in options.values()), 'unsupported restoration option representation')
    return options


def verify_stage_scope(root, native, scope, witnesses, production_options):
    """Bind each file to an executed stage, never to the union of all runtimes."""
    stages = scope.get('stages')
    require(isinstance(stages, dict) and set(stages) == set(STAGE_COMPONENTS),
            'stage-specific shared implementation scope required')
    for stage, required in STAGE_COMPONENTS.items():
        declared = stages[stage]; observed = witnesses[stage]
        require(declared.get('evidence') == observed['evidence'], 'stage evidence differs: '+stage)
        components = declared.get('component_files', [])
        require(len(components) == len(required) and {c['relative_path'] for c in components} == required,
                'stage implementation coverage incomplete: '+stage)
        require(observed['runtime_hashes'], 'stage has no executed runtime: '+stage)
        for item in components:
            relative = Path(item['relative_path']); witness = _reference(root, item['validated_implementation'])
            witness_root = witness
            for _ in relative.parts: witness_root = witness_root.parent
            require(witness_root.is_relative_to(Path(root)/'code') and
                    file_digest(witness_root/'.snapshot_commit') in observed['runtime_hashes'],
                    'component did not execute this stage: '+stage)
            actual = (native/relative).resolve()
            require(actual.is_relative_to(native) and file_digest(actual) == file_digest(witness),
                    'production component differs from tested stage: '+stage)
        if stage == 'restoration':
            require(declared.get('effective_options') == production_options and
                    all(options == production_options for options in observed['effective_options']),
                    'production restoration options differ from executed controls')


def verify_applicable_pool(root, reference, own, exact_reference, started_at):
    """The actual selected batch must include this exact production canary."""
    scope = checked(root, reference)
    require(scope.get('schema') == 'isg-training-applicable-pool-v1' and
            scope.get('config_sha256') == own['source_proof']['config']['sha256'],
            'exact configuration lacks applicable selected-pool evidence')
    evidence = checked(root, scope['execution'])
    require(exact_reference in evidence.get('canaries', []),
            'exact production canary did not execute the applicable pool')
    manifest_path = _reference(root, scope['pool_manifest']); manifest = checked(root, scope['pool_manifest'])
    config = checked(root, own['source_proof']['config']); template = config['template']
    require(scope.get('batch_id') == manifest.get('batch_id') and scope.get('selected_scope') == manifest.get('selected_scope')
            and scope.get('batch_id') and scope.get('selected_scope'), 'selected batch identity missing or different')
    from ispatialgym.sampling.model_pool import read_manifest
    actual, actual_ref = read_manifest(str(manifest_path), scene=template['scene'], rooms=[template['room']],
                                      fixed_bindings=manifest['fixed_bindings'])
    require(actual_ref == scope['pool_manifest'], 'selected pool manifest differs')
    selection = checked(root, manifest['selection_proof'])
    require(scope.get('selected_configuration_id') in selection.get('configuration_ids', []),
            'pool selection does not cover the exact selected configuration')
    target = [o for o in template['objects'] if o['role'] == 'target']
    require(len(target) == 1, 'applicable pool verifier requires a supported exact target contract')
    target = target[0]; scale = config['perturbations']['scales'][target['instance']]['applied']
    matches = [e for e in actual['entries'] if (e['category'], e['model'], e['scale']) ==
               (target['category'], target['model'], scale)]
    require(len(matches) == 1, 'selected model and applied scale absent from tested pool')
    result = checked(root, evidence['sequence_result']); request = checked(root, result['request'])
    from .training_pipeline_status import sequence_stage_references
    for stage in result['stages']:
        refs = sequence_stage_references(root, Path(evidence['sequence_result']['path']), stage)
        require(checked(root, refs['pool_execution'])['manifest'] == scope['pool_manifest'],
                'applicable execution journal belongs to another selected pool')
    for row in request['records']:
        path = Path(row['environment']['ISPATIALGYM_CAMPAIGN_MODEL_POOL_MANIFEST']).resolve()
        require(path == manifest_path and file_digest(path) == scope['pool_manifest']['sha256'],
                'applicable sampler activated another pool')
    verified = verify_sampler_pool(root, scope['execution'], started_at)
    return {'reference': reference, 'pool_manifest': scope['pool_manifest'], 'batch_id': scope['batch_id'],
            'selected_configuration_id': scope['selected_configuration_id'], 'verification': verified}


def verify_configuration(root, config_path, reference):
    root = Path(root).resolve(); config_path = Path(config_path).resolve()
    require(config_path.is_relative_to(root), 'configuration outside campaign')
    config = json.loads(config_path.read_text()); release = checked(root, reference)
    require(release.get('schema') == 'isg-training-configuration-release-v2' and
            release.get('campaign_id') == 'campaign_20261110', 'exact scoped release missing')
    require(release['config_sha256'] == file_digest(config_path) and
            config['split'] == config['template']['split'] == 'train' and
            config.get('purpose') == 'training_production', 'release configuration is not exact production input')
    started = float(release['campaign_started_at']); task = config['template']['task']; key = config['validation_key']
    own = validated_canary(root, release['exact_canary'], started)
    require(own['source_proof']['config']['sha256'] == release['config_sha256'] and
            own['source_proof']['validation_key'] == key, 'exact production configuration lacks its own canary')
    verify_render(root, release['exact_render'], own)
    applicable_pool = verify_applicable_pool(root, release['applicable_pool'], own, release['exact_canary'], started)
    path = checked(root, release['applicable_path'])
    require(path.get('schema') == 'isg-training-applicable-path-v1' and path.get('task') == task,
            'another family cannot validate this task path')
    require(release['exact_canary'] in path['canaries'] and path['canaries'], 'applicable path omits exact configuration')
    for ref in path['canaries']:
        case = validated_canary(root, ref, started)
        require(case['task'] == task and case['export_record']['validation_key']['code_sha256'] == key['code_sha256'],
                'applicable path runtime/task differs')
    shared = checked(root, release['shared_pipeline'])
    require(shared.get('schema') == 'isg-training-shared-pipeline-v2', 'shared executed pipeline proof missing')
    cases = [validated_canary(root, ref, started) for ref in shared['augmentation_cases']]
    perturbations = [case['export_record']['perturbations'] for case in cases]
    require({bool(p['lighting']) for p in perturbations} == {False, True} and
            {bool(p['removed_objects']) for p in perturbations} == {False, True}, 'default/applied augmentation paths untested')
    require(any(any(s['default'] != s['applied'] for s in p['scales'].values()) for p in perturbations),
            'actual nondefault scale execution missing')
    controls = verify_controls(root, shared['restoration_controls'], started)
    rendering = verify_batch(root, shared['batch_rendering'], started, 'kinematic_observation_rendering')
    executions = {name: verify_probe_execution(root, ref, ref['execution'])
                  for name, ref in shared['restoration_controls'].items()}
    executions['batch_rendering'] = verify_probe_execution(
        root, shared['batch_rendering'], shared['batch_rendering']['execution'])
    pool = verify_sampler_pool(root, shared['sampler_pool'], started)
    # Bind shared implementation identities explicitly. A task-local restoration
    # option cannot silently validate every other path or runtime.
    scope = checked(root, release['shared_implementation_scope'])
    require(scope.get('schema') == 'isg-training-shared-implementation-scope-v1' and
            scope.get('validation_key') == key and scope.get('shared_pipeline') == release['shared_pipeline'],
            'shared implementation applicability to this exact runtime is unproved')
    native = Path(release['native_runtime']).resolve()
    require(native.is_relative_to(root/'code') and file_digest(native/'.snapshot_commit') == key['code_sha256'],
            'production runtime differs')
    control_probes = [checked(root, ref) for ref in shared['restoration_controls'].values()]
    control_options = []
    for probe in control_probes:
        for entry in probe['entries']:
            case = validated_canary(root, entry['canary_review'], started)
            options = restoration_options(root, case)
            report = checked(root, entry['report']); loader = report['loader']
            for option, field in [('serialized_mass_before_reset_v1', 'pre_reset_serialized_mass'),
                                  ('pristine_mass_before_reset_v1', 'pre_reset_pristine_mass')]:
                require(not options.get(option) or loader.get(field, {}).get('revision') == option,
                        'control did not execute its configured mass restoration')
            require(not options.get('serialized_root_motion_v1') or
                    loader['campaign_restoration'].get('serialized_root_motion', {}).get('revision') == 'serialized_root_motion_v1',
                    'control did not execute its configured motion restoration')
            control_options.append(options)
    pool_evidence = checked(root, shared['sampler_pool'])
    pool_result = checked(root, pool_evidence['sequence_result'])
    pool_request = checked(root, pool_result['request'])
    pool_hashes = {file_digest(Path(row['native_runtime'])/'.snapshot_commit') for row in pool_request['records']}
    render_probe = checked(root, shared['batch_rendering'])
    witnesses = {
        'restoration': {'evidence': shared['restoration_controls'],
                        'runtime_hashes': {p['probe_runtime']['sha256'] for p in control_probes},
                        'effective_options': control_options},
        'sampler_pool': {'evidence': shared['sampler_pool'], 'runtime_hashes': pool_hashes},
        'rendering': {'evidence': shared['batch_rendering'],
                      'runtime_hashes': {render_probe['probe_runtime']['sha256']}},
    }
    verify_stage_scope(root, native, scope, witnesses, restoration_options(root, own))
    options = restoration_options(root, own)
    cold = None
    if options.get('pristine_mass_before_reset_v1') or options.get('serialized_mass_before_reset_v1'):
        require(shared.get('cold_reset_compatibility'), 'original cold-reset compatibility evidence missing')
        cold = verify_cold_reset_compatibility(root, shared['cold_reset_compatibility'], started, native, task)
        cold_evidence = checked(root, shared['cold_reset_compatibility'])
        executions['cold_candidate'] = verify_probe_execution(
            root, cold_evidence['candidate'], cold_evidence['candidate_execution'])
    from .execution_dependencies import verify_execution_dependencies
    dependencies = verify_execution_dependencies(root, release['config_sha256'], 'native_generation')
    require(dependencies['status'] == 'exact_originals_checked' and dependencies['dependencies'],
            'exact materialized configuration dependency chain is incomplete')
    expected = {ref['sha256'] for ref in release['dependencies']}
    require(expected and expected == {d['artifact']['sha256'] for d in dependencies['dependencies']},
            'release dependencies differ from connected verified originals')
    for ref in release['dependencies']: _reference(root, ref)
    return {'release': reference, 'config_sha256': release['config_sha256'], 'validation_key': key,
            'task': task, 'class_id': config['template']['class_id'], 'controls': controls,
            'rendering': rendering, 'sampler_pool': pool, 'applicable_pool': applicable_pool, 'dependencies': dependencies,
            'cold_reset_compatibility': cold,
            'native_process_executions': executions,
            'status': 'exact_evidence_verified', 'production_authorization': False}


def verify_diversity_cell(root, config_path, ticket_ref):
    """Opaque planning IDs must be tied to exact physical configuration facts."""
    ticket = checked(root, ticket_ref); plan = checked(root, ticket['plan'])
    cells = [c for c in plan['cells'] if c['id'] == ticket['cell_id']]
    require(len(cells) == 1, 'ticket cell missing or duplicated')
    cell = cells[0]; config = json.loads(Path(config_path).read_text()); template = config['template']
    require(file_digest(config_path) == ticket['config_sha256'] and
            ticket['config_sha256'] in cell.get('configuration_sha256s', []),
            'exact configuration is not assigned to this diversity cell')
    supports = sorted(o['instance'] for o in template['objects'] if o['role'] == 'support')
    require(supports, 'diversity support inventory missing')
    support_id = supports[0] if len(supports) == 1 else 'support_set:'+digest(supports)
    require(cell['support_id'] == support_id and
            cell.get('support_instances', supports if len(supports) == 1 else None) == supports,
            'diversity named support set differs')
    model_keys = sorted({o['category']+'/'+o['model'] for o in template['objects']
                         if o.get('semantic_kind') not in ('floor', 'wall')})
    require(sorted(cell['model_keys']) == model_keys and
            (cell['class_id'], cell['scene'], cell['room']) ==
            (template['class_id'], template['scene'], template['room']), 'diversity model/location differs')
    physical = {'objects': [{k:o[k] for k in ('instance','role','category','model','asset_sha256','default_scale')}
                            for o in sorted(template['objects'],key=lambda o:o['instance'])],
                'applied_scales': {k:v['applied'] for k,v in sorted(config['perturbations']['scales'].items())},
                'scene_setup_sha256': digest(template['scene_setup'])}
    require(cell.get('physical_identity') == physical,
            'diversity model/scale/measured support context differs from configuration')
    return {'cell_id':cell['id'], 'physical_identity_sha256':digest(physical)}


def check_production(root, config_path, environment, policy, *, start, count):
    """Current registry, complete canonical chain, exact evidence and quota ticket."""
    from .production_gate import canonical_training_ready
    from .training_diversity import check_ticket
    require(type(start) is int and start > 0 and type(count) is int and count > 0, 'production range missing')
    require(canonical_training_ready(root), 'all100 exact canonical classes required for bulk production')
    reference = {'path': environment['ISPATIALGYM_TRAINING_RELEASE'],
                 'sha256': environment['ISPATIALGYM_TRAINING_RELEASE_SHA256']}
    require(policy.get('mode') == 'scoped_releases' and policy.get('standalone_training_allowed') is True,
            'bulk production remains held')
    config_sha = file_digest(config_path)
    matches = [r for r in policy.get('configuration_releases', []) if r.get('config_sha256') == config_sha]
    require(matches == [{'config_sha256': config_sha, 'release': reference}], 'exact release is not registered once')
    assert_production_range(root, start, count, policy)
    proof = verify_configuration(root, config_path, reference)
    ticket = {'path': environment['ISPATIALGYM_TRAINING_DIVERSITY_TICKET'],
              'sha256': environment['ISPATIALGYM_TRAINING_DIVERSITY_TICKET_SHA256']}
    capacity = check_ticket(root, config_sha, start, count, ticket)
    require((capacity['task'], capacity['class_id']) == (proof['task'], proof['class_id']), 'ticket scope differs')
    diversity = verify_diversity_cell(root, config_path, ticket)
    return {**proof, 'training_diversity_ticket': ticket, 'capacity': capacity, 'diversity': diversity, 'production_authorization': True}


def assert_production_range(root, start, count, policy):
    """Original canary request ranges remain zero-credit after policy revisions."""
    ranges = list(policy.get('allowed_canaries', [])) + list(policy.get('allowed_existing_canary_reviews', []))
    for path in (Path(root)/'training_pipeline_canaries').glob('*/requests*.json'):
        value = json.loads(path.read_text())
        require(value.get('campaign_id') == 'campaign_20261110', 'foreign canary range inventory')
        ranges.extend(value.get('records', []))
    for row in ranges:
        require(type(row.get('start')) is int and type(row.get('count')) is int and row['count'] > 0,
                'canary range inventory incomplete')
        require(not (start < row['start']+row['count'] and row['start'] < start+count),
                'zero-credit canary range cannot become production')


def acceptance_proof(root, review):
    """Recheck current release before a normal canonical group transaction."""
    from .production_gate import read_environment, read_policy
    from .training_policy_history import preserve
    from .contracts import write_json_atomic
    job = Path(review['source_job']); environment = read_environment(job/'job.env')
    policy_path = Path(root)/'controllers/training_production_gate.json'
    payload = policy_path.read_bytes(); policy, sha = read_policy(root)
    require(file_digest(policy_path) == sha and digest(json.loads(payload)) == digest(policy), 'policy changed while reading')
    config_path = Path(environment['ISPATIALGYM_CAMPAIGN_CONFIG'])
    start, count = int(environment['START']), int(environment['N'])
    proof = check_production(root, config_path, environment, policy, start=start, count=count)
    require(proof['validation_key'] == review['validation_key'], 'production member source differs from release')
    members = [r['instance_id'] for r in review['members']]
    require(all(start <= int(i.rsplit('__',1)[-1]) < start+count for i in members), 'production member outside issued ticket')
    current, _ = read_policy(root)
    require(current.get('mode') == 'scoped_releases' and current.get('standalone_training_allowed') is True and
            current.get('configuration_releases') == policy.get('configuration_releases'), 'release revoked during acceptance')
    receipt = {'schema': 'isg-training-production-claim-proof-v1', 'campaign_id': 'campaign_20261110',
        'release': proof['release'], 'config': {'path': str(config_path), 'sha256': proof['config_sha256']},
        'ticket': proof['training_diversity_ticket'], 'validation_key': review['validation_key'],
        'review_sha256': digest(review), 'source_job': str(job), 'instance_ids': members,
        'policy': preserve(root, payload), 'verification': proof, 'benchmark_credit_only_after_canonical_commit': True}
    path = Path(root)/'validation/training_production/claims'/(digest(receipt)+'.json')
    if path.exists(): require(json.loads(path.read_text()) == receipt, 'immutable production proof changed')
    else: write_json_atomic(path, receipt)
    return {'training_config_path': str(config_path), 'training_config_sha256': proof['config_sha256'],
            'training_release': proof['release'], 'training_diversity_ticket': proof['training_diversity_ticket'],
            'training_production_receipt': {'path': str(path), 'sha256': file_digest(path)}}


def accepted_record_identity(root, row):
    """Immutable committed membership, without live-budget/quota recursion."""
    from .training_diversity import check_ticket_identity
    receipt = checked(root, row['training_production_receipt'])
    require(receipt.get('schema') == 'isg-training-production-claim-proof-v1' and
            receipt['campaign_id'] == 'campaign_20261110' and row['instance_id'] in receipt['instance_ids'],
            'accepted production membership unproved')
    require(receipt['release'] == row['training_release'] and receipt['ticket'] == row['training_diversity_ticket']
            and receipt['config'] == {'path': row['training_config_path'], 'sha256': row['training_config_sha256']}
            and receipt['validation_key'] == row['validation_key'], 'accepted release/ticket identity differs')
    cfg = checked(root, receipt['config']); release = checked(root, receipt['release'])
    require(cfg.get('purpose') == 'training_production' and cfg['validation_key'] == row['validation_key']
            and release['config_sha256'] == row['training_config_sha256'], 'accepted configuration changed')
    policy = checked(root, receipt['policy'])
    require(policy.get('mode') == 'scoped_releases' and policy.get('standalone_training_allowed') is True and
            {'config_sha256': row['training_config_sha256'], 'release': row['training_release']}
            in policy.get('configuration_releases', []), 'historical exact release authorization missing')
    ticket = checked(root, receipt['ticket'])
    check_ticket_identity(root, row['training_config_sha256'], ticket['start'], ticket['count'], receipt['ticket'])
    index = int(row['instance_id'].rsplit('__',1)[-1])
    require(ticket['start'] <= index < ticket['start']+ticket['count'] and
            (ticket['class_id'], ticket['task']) == (row['class_id'], row['task']), 'accepted production exceeds ticket scope')
    assert_production_range(root, index, 1, policy)
    return receipt


def bind_refill(root, queue_path, command, *, release, plan, cell_id, request_id, controller_path, producer_state, owner):
    """Authorize one prepared exact queue; caller retains ordinary enqueue/observer ownership.

    Index reservation and physical authoring happen through the existing native
    family adapter. This operation neither invents a configuration nor dispatches.
    """
    from .production_gate import _inputs, read_policy, check_dispatch
    from .training_diversity import reserve_ticket, bind_ticket_controller
    from .contracts import write_json_atomic
    producer_state = Path(producer_state).resolve()
    require(producer_state.is_relative_to(Path(root).resolve()/'controllers') and producer_state.is_file(),
            'refill needs its existing exact producer state')
    queue = json.loads(Path(queue_path).read_text()); config_path = _reference(root, {
        'path': queue['config_path'], 'sha256': queue['config_sha256']})
    config = json.loads(config_path.read_text())
    require(config.get('purpose') == 'training_production', 'refill needs its exact production config')
    proof = verify_configuration(root, config_path, release)
    # The normal command carries authoritative allocation range. Require its
    # queue/config binding before reserving any diversity capacity.
    environment = {}
    for index, word in enumerate(command):
        if word == '--extra-env':
            key, value = command[index+1].split('=', 1)
            require(key not in environment or environment[key] == value, 'conflicting refill environment')
            environment[key] = value
    actual_path, _, actual_queue = _inputs(root, environment)
    require(actual_path == config_path and actual_queue == queue, 'refill command/queue differs')
    start = int(command[command.index('--base-index')+1]); count = int(command[command.index('submit')+2])
    policy, _ = read_policy(root)
    from .production_gate import canonical_training_ready
    require(canonical_training_ready(root) and policy.get('mode') == 'scoped_releases' and
            policy.get('standalone_training_allowed') is True, 'bulk production remains held')
    require({'config_sha256': proof['config_sha256'], 'release': release} in policy.get('configuration_releases', []),
            'refill configuration has no registered release')
    assert_production_range(root, start, count, policy)
    ticket = reserve_ticket(root, request_id, proof['config_sha256'], start, count, cell_id, plan['path'], plan['sha256'])
    verify_diversity_cell(root, config_path, ticket)
    bound = list(command)
    fields = {'ISPATIALGYM_TRAINING_RELEASE': release['path'], 'ISPATIALGYM_TRAINING_RELEASE_SHA256': release['sha256'],
              'ISPATIALGYM_TRAINING_DIVERSITY_TICKET': ticket['path'],
              'ISPATIALGYM_TRAINING_DIVERSITY_TICKET_SHA256': ticket['sha256']}
    for key, value in fields.items():
        require(key not in environment, 'refill command already carries a release binding')
        bound += ['--extra-env', key+'='+value]
    decision = check_dispatch(root, bound)
    require(decision['allowed'] and decision['category'] == 'released_training', 'prepared refill failed current gate')
    controller_path = Path(controller_path).resolve()
    require(controller_path.is_relative_to(Path(root).resolve()/'controllers') and owner,
            'refill needs a named campaign controller owner')
    controller = {'schema':'isg-training-ticket-controller-v1', 'campaign_id':'campaign_20261110',
        'owner':owner, 'ticket':ticket, 'config_sha256':proof['config_sha256'], 'start':start, 'count':count,
        'phase':'prepared', 'jobs':[],
        'pending_dispatch':[{'queue':str(Path(queue_path).resolve()), 'start':start, 'count':count}],
        'pending_acceptance':[{'start':start, 'count':count}],
        'outcome_artifacts':[], 'queue':{'path':str(Path(queue_path).resolve()),'sha256':file_digest(queue_path)},
        'command':bound, 'command_sha256':digest(bound),
        'next_action':'Dispatch only through current guarded router; retain actual job/replay/claim outcomes before finalization.'}
    if controller_path.exists(): require(json.loads(controller_path.read_text()) == controller, 'refill controller already advanced or differs')
    else: write_json_atomic(controller_path, controller)
    bind_ticket_controller(root, ticket, controller_path)
    from .training_ticket_observer import register
    registration = register(root, ticket, producer_state)
    return {'command': bound, 'ticket': ticket, 'release': release, 'decision': decision,
            'start': start, 'count': count, 'controller':{'path':str(controller_path),'sha256':file_digest(controller_path)},
            'terminal_observer': registration, 'dispatched': False}
