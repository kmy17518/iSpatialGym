"""Strict, read-only audit of changed-augmentation one-world restoration.

Independent references, ordinary success, ordinary failure and interruption are
separate witnesses. This audit neither dispatches nor releases production.
"""
from pathlib import Path

from .contracts import digest, file_digest, require
from .acceptance import report_passed
from .training_restoration import checked, validated_canary, same_loaded_state
from .training_batch_probe import inputs, verify_runtime_compatibility
from .training_release import _observation, same_reset_boundary, verify_probe_execution

PHYSICS = {'fresh_A': ('FRESH_A', [0]), 'fresh_B': ('FRESH_B', [1]),
           'successful_AA': ('A_A', [0, 0]), 'failed_AA': ('A_FAILURE_A', [0, 0]),
           'interrupted_AA': ('A_INTERRUPT_A', [0, 0]), 'ABA': (None, [0, 1, 0])}
RENDER = {k: PHYSICS[k] for k in ('fresh_A', 'fresh_B', 'ABA')}


def one_world(probe, observations):
    construction = probe.get('world_construction', {})
    require(construction.get('attempted') == construction.get('completed') == 1 and
            construction.get('method') == 'omnigibson.Environment',
            'exact one-world constructor observation missing')
    require(observations and len({o['process_session'] for o in observations}) == 1 and
            len({o['scene_runtime_id'] for o in observations}) == 1 and
            len({o['sim_runtime_id'] for o in observations}) == 1, 'sequence rebuilt its native world')
    require(all(a['sequence'] < b['sequence'] for a, b in zip(observations, observations[1:])),
            'sequence observation order differs')
    return observations[0]['process_session']


def compare_physics_reference(reference, reused):
    """Keep the original reset, final pre-action and first action tolerances."""
    for a, b in zip(reference['boundaries'], reused['boundaries']):
        same_reset_boundary(a, b)
    same_loaded_state(reference['boundaries'][-1]['loaded_state'], reused['boundaries'][-1]['loaded_state'])
    same_reset_boundary(reference['step']['after'], reused['step']['after'])


def verify(root, reference, started_at):
    document = checked(root, reference)
    require(document.get('schema') == 'isg-sequential-augmentation-verification-input-v1' and
            set(document['physics']) == set(PHYSICS) and set(document['rendering']) == set(RENDER) and
            set(document['execution']) == {'physics', 'rendering'} and
            set(document['execution']['physics']) == set(PHYSICS) and
            set(document['execution']['rendering']) == set(RENDER),
            'complete independent-reference/success/failure/interruption matrix required')
    sources = {}; source_refs = None; runtime = None; artifacts = set(); results = {}; all_processes = set()
    for stage, specifications in (('physics', PHYSICS), ('rendering', RENDER)):
        cases = {}; stage_results = {}
        expected_kind = 'physics_action_open_loop' if stage == 'physics' else 'kinematic_observation_rendering'
        for name, (control, order) in specifications.items():
            probe_ref = document[stage][name]; probe = checked(root, probe_ref)
            execution=verify_probe_execution(root,probe_ref,document['execution'][stage][name])
            request_ref = probe['request']; checked(root, request_ref)
            request, records = inputs(root, request_ref['path'])
            require(request.get('sequential_augmentation') is True and not request.get('physical_candidate') and
                    request.get('diagnostic_control') == control and probe['kind'] == expected_kind and
                    len(probe['entries']) == len(order), 'sequential reference/control identity differs')
            source_refs = source_refs or request['canaries']
            require(request['canaries'] == source_refs, 'sequential controls use different validated inputs')
            identity = probe['probe_runtime']; runtime = runtime or identity
            require(identity == runtime and file_digest(identity['path']) == request['probe_runtime_sha256'] == identity['sha256'],
                    'sequential matrix runtime differs')
            require(verify_runtime_compatibility(root, request, records, Path(identity['path']).parent) ==
                    probe['runtime_compatibility'], 'sequential matrix runtime proof differs')
            observations = []; case = []
            for ordinal, (entry, member) in enumerate(zip(probe['entries'], order)):
                require(entry['canary_review'] == source_refs[member], 'sequential member order differs')
                key = digest(entry['canary_review'])
                if key not in sources: sources[key] = validated_canary(root, entry['canary_review'], started_at)
                source = sources[key]; exported = source['export_record']; report = checked(root, entry['report'])
                if stage == 'physics':
                    require(report['instance_id'] == source['instance_id'] and Path(report['replay']).resolve() ==
                            Path(exported['artifacts']['demo']['path']).resolve(), 'sequential physics source differs')
                    failure = ordinal == 0 and name in ('failed_AA', 'interrupted_AA')
                    if failure:
                        exception = 'ControlledEpisodeFailure' if name == 'failed_AA' else 'ControlledEpisodeInterrupt'
                        abort = report['loader']['campaign_restoration']['controlled_abort']
                        require(entry.get('expected_controlled_abort') is True and entry['returncode'] != 0 and
                                report.get('error', '').startswith(exception+'(') and abort == {
                                    'native_control_calls_completed': 1, 'physics_steps_added': 0,
                                    'exception_type': exception, 'injected': True}, 'native failure/interruption cleanup was not exercised')
                        require(not report.get('recorder_error') and not entry.get('result', {}).get('meta_error'),
                                'interrupted native episode did not finish ordinary evidence cleanup')
                    else:
                        report_passed(report, 'replay'); require(entry['returncode'] == 0, 'sequential replay failed')
                    require(bool(report['loader'].get('reused_world')) == (ordinal > 0),
                            'sequential native build/reuse history differs')
                    restoration = report['loader']['campaign_restoration']
                    boundaries = [_observation(report, key) for key in ('native_reset_boundary_observation',
                        'native_reset_step_completed_observation', 'native_final_pre_action_observation')]
                    step = restoration['next_native_control_step_observation']
                    require(step['native_control_calls_observed'] == 1 and step['physics_steps_added'] == 0,
                            'first unchanged native action observation missing')
                    for endpoint in ('before', 'after'):
                        require(step[endpoint]['loaded_state_sha256'] == digest(step[endpoint]['loaded_state']),
                                'first native action state changed')
                    same_loaded_state(boundaries[-1]['loaded_state'], step['before']['loaded_state'])
                    observation = boundaries[-1]
                    case.append({'member': member, 'boundaries': boundaries, 'step': step, 'report': entry['report']})
                else:
                    require(report.get('tag') == 'REPLAY_DEMO' and report.get('kept') is True and
                            Path(report['demo']).name == source['instance_id'] and isinstance(report.get('episode_index'), int),
                            'sequential rendering episode failed')
                    restoration = report['campaign_restoration']; observation = restoration['native_observation']
                    require(entry.get('outputs'), 'sequential rendering output missing')
                    for output in entry['outputs']:
                        identity = (output['path'], output['sha256'])
                        if identity in artifacts: continue
                        path = Path(output['path']).resolve()
                        require(path.is_relative_to(Path(root).resolve()) and path.stat().st_size > 0 and
                                file_digest(path) == output['sha256'], 'sequential rendered artifact changed')
                        artifacts.add(identity)
                    case.append({'member': member, 'observation': observation, 'report': entry['report']})
                require(restoration.get('scale_verified') is True and observation.get('scheduler_job_id') and
                        observation.get('simulator_module', '').startswith('omnigibson.') and
                        observation['validation_key'] == exported['validation_key'] and
                        observation['perturbations_sha256'] == digest(exported['perturbations']) and
                        observation['loaded_state_sha256'] == digest(observation['loaded_state']),
                        'sequential native observation binding differs')
                observations.append(observation)
            process = one_world(probe, observations)
            require(process not in all_processes, 'independent reference/control reused another process')
            all_processes.add(process); cases[name] = case
            stage_results[name] = {'probe': probe_ref, 'process_session': process, 'world_load_count': 1,
                                   'execution':execution,
                                   'reports': [row['report'] for row in case]}
        for name, case in cases.items():
            for episode in case:
                independent = cases['fresh_A' if episode['member'] == 0 else 'fresh_B'][0]
                if stage == 'physics': compare_physics_reference(independent, episode)
                else: same_loaded_state(independent['observation']['loaded_state'], episode['observation']['loaded_state'])
        results[stage] = stage_results
    return {'schema': 'isg-sequential-augmentation-verification-v1', 'input': reference,
            'status': 'verified', 'runtime': runtime, 'sources': source_refs, 'stages': results,
            'independent_processes': len(all_processes), 'checks': ['changed lighting and fixed-actor parking',
                'one actual Environment construction per process', 'exact native task/model/scale/room/assets',
                'independent A and B references', 'A/B/A restoration', 'successful A/A cleanup',
                'failed A/A cleanup', 'interrupted A/A cleanup', 'unchanged reset/first-action tolerances',
                'exact rendered artifacts'], 'production_released': False, 'benchmark_credit': 0}
