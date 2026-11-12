"""Exact retained-source loading boundaries, with explicit fresh-world accounting.

This module does not authorize generation, production release, or acceptance.
"""
import json
from pathlib import Path

from .contracts import digest, file_digest, require

ORDERS = {'FRESH_A': [0], 'FRESH_B': [1], 'AA': [0, 0], 'ABA': [0, 1, 0]}


def ref(path):
    return {'path': str(path), 'sha256': file_digest(path)}


def checked(root, reference):
    path = Path(reference['path']).resolve()
    require(path.is_relative_to(Path(root).resolve()) and file_digest(path) == reference['sha256'],
            'loading-boundary artifact changed or escaped campaign')
    return path


def segment_plan(request):
    require(request.get('schema') == 'isg-retained-loading-boundary-request-v1'
            and request.get('campaign_id') == 'campaign_20261110'
            and request.get('boundary_kind') == 'room'
            and request.get('control') in ORDERS
            and request.get('production_released') is False
            and type(request.get('benchmark_credit')) is int and request['benchmark_credit'] == 0
            and type(request.get('timeout_seconds')) is int and 0 < request['timeout_seconds'] <= 1800,
            'invalid or unbounded loading-boundary request')
    order = ORDERS[request['control']]
    require(request.get('max_native_episodes') == len(order) and len(request['sources']) == 2,
            'loading-boundary native episode budget differs')
    if request['control'] == 'AA':
        return [{'source': 0, 'episodes': 2, 'request': request['compatible_AA_request']}]
    return [{'source': index, 'episodes': 1, 'request': request['sources'][index]} for index in order]


def inputs(root, request_path, *, runtime=None, source_reader=None):
    from .training_mechanism_probe import inputs as mechanism_inputs
    source_reader = source_reader or mechanism_inputs
    root = Path(root).resolve(); path = Path(request_path).resolve()
    require(path.is_relative_to(root), 'loading-boundary request outside campaign')
    q = json.loads(path.read_text()); plan = segment_plan(q)
    runtime = Path(runtime).resolve() if runtime else Path(__file__).resolve().parents[2]
    require(ref(runtime/'.snapshot_commit') == q['native_runtime'], 'loading-boundary runtime differs')
    sources = []
    for reference in q['sources']:
        request, row, environment = source_reader(root, checked(root, reference), runtime=runtime)
        require(request['control'] == 'FRESH_A' and request['native_runtime'] == q['native_runtime']
                and request.get('idempotent_fixed_pose_sets') is None,
                'loading-boundary source is not the exact approved cold configuration')
        directory = Path(row['source_tro']['path']).parent.parent
        metadata = json.loads((directory/'template_meta.json').read_text())
        sources.append({'request': request, 'row': row, 'environment': environment,
                        'room': {key: metadata.get(key) for key in ('scene', 'load_room_instances')}})
    require(sources[0]['row']['instance_id'] != sources[1]['row']['instance_id']
            and sources[0]['room'] != sources[1]['room'], 'room control needs distinct retained worlds')
    require(sources[0]['request']['physical_delta'] == sources[1]['request']['physical_delta'],
            'loading-boundary physical execution settings differ')
    if q['control'] == 'AA':
        request, row, _ = source_reader(root, checked(root, q['compatible_AA_request']), runtime=runtime)
        require(request['control'] == 'AA' and row == sources[0]['row']
                and request['native_runtime'] == q['native_runtime']
                and request['physical_delta'] == sources[0]['request']['physical_delta'],
                'compatible control source or execution settings differ')
    return q, sources, plan


def native_boundary(runner, current_tro, incoming_tro):
    """Call the actual native queue compatibility APIs without modifying a world."""
    signatures = [runner.instance_signature(p) for p in (current_tro, incoming_tro)]
    directories = [runner.resolve_instance(p)[1] for p in (current_tro, incoming_tro)]
    compatible, reason = runner.template_compatible(directories[0], current_tro,
                                                   directories[1], incoming_tro)
    reuse = signatures[0] == signatures[1] and compatible
    return {'method': 'native instance_signature + template_compatible',
            'signatures': signatures, 'template_compatible': compatible, 'reason': reason,
            'reuse_allowed': reuse, 'required_returncode': 0 if reuse else runner.EXIT_QUEUE_RESTART,
            'physical_steps_added': 0, 'world_mutations': 0}


def validate_sequence(request, observation, children):
    """Reject inferred reuse, incomplete children, wrong sources and hidden reloads."""
    plan = segment_plan(request)
    require(observation.get('schema') == 'isg-retained-loading-boundary-observation-v1'
            and observation.get('status') == 'completed'
            and observation.get('native_runtime') == request['native_runtime']
            and observation.get('benchmark_credit') == 0
            and observation.get('production_released') is False
            and len(observation['segments']) == len(children) == len(plan),
            'loading-boundary sequence incomplete or runtime differs')
    pids = []; worlds = 0; episodes = 0
    for ordinal, (expected, segment, child) in enumerate(zip(plan, observation['segments'], children)):
        require(segment['ordinal'] == ordinal and segment['source'] == expected['source']
                and child['request'] == expected['request']
                and child['status'] == 'native_mechanism_controls_completed'
                and child['native_runtime'] == request['native_runtime']
                and len(child['entries']) == expected['episodes']
                and child['world_construction'] == {'attempted': 1, 'completed': 1},
                'loading-boundary child source, completion or world count differs')
        require(all(e['returncode'] == 0 and e['result'].get('success') is True
                    and e.get('expected_abort') is False for e in child['entries']),
                'loading-boundary child action failed')
        pids.append(segment['process_pid']); worlds += 1; episodes += len(child['entries'])
        if ordinal+1 < len(plan):
            boundary = segment['boundary']
            require(segment['returncode'] == 75 and boundary['required_returncode'] == 75
                    and boundary['reuse_allowed'] is False
                    and boundary['physical_steps_added'] == boundary['world_mutations'] == 0
                    and (boundary['signatures'][0] != boundary['signatures'][1]
                         or boundary['template_compatible'] is False),
                    'fresh-world transition lacks an actual incompatible native decision')
        else:
            require(segment['returncode'] == 0 and segment['boundary'] is None,
                    'final loading segment did not finish normally')
    require(len(set(pids)) == len(pids) and all(type(pid) is int and pid > 0 for pid in pids),
            'fresh-process boundary was not observed')
    require(episodes == request['max_native_episodes'] and observation['world_loads'] == worlds
            and observation['native_episodes'] == episodes, 'loading-boundary accounting differs')
    return {'world_loads': worlds, 'native_episodes': episodes, 'process_pids': pids,
            'one_world_reuse': request['control'] == 'AA',
            'room_changes': max(0, worlds-1), 'benchmark_credit': 0, 'production_released': False}


def report_case(report, job_id):
    from .acceptance import report_passed
    from .training_release import _observation
    from .training_restoration import same_loaded_state
    report_passed(report, 'replay')
    snapshots = report['loader']['complete_native_snapshot_inventory']
    require(snapshots and all(v['complete'] and v['physics_steps_added'] == 0 for v in snapshots),
            'complete native snapshot proof missing')
    boundaries = [_observation(report, key) for key in ('native_reset_boundary_observation',
                   'native_reset_step_completed_observation', 'native_final_pre_action_observation')]
    step = report['loader']['campaign_restoration']['next_native_control_step_observation']
    require(step['native_control_calls_observed'] == 1 and step['physics_steps_added'] == 0,
            'native first-action observation missing')
    for value in [*boundaries, step['before'], step['after']]:
        require(str(value['scheduler_job_id']) == str(job_id)
                and value['loaded_state_sha256'] == digest(value['loaded_state']),
                'native loading observation provenance changed')
    same_loaded_state(boundaries[-1]['loaded_state'], step['before']['loaded_state'])
    return {'boundaries': boundaries, 'step': step}


def executed_job(root, observation, execution):
    submission = json.loads(checked(root, execution['submission']).read_text())
    job = str(submission['job_id'])
    require(str(observation['scheduler_job_id']) == job, 'loading-boundary allocation identity differs')
    lines = checked(root, execution['scheduler']).read_text().splitlines()
    scheduler = [line.split('|') for line in lines if line.split('|')[0] == job]
    require(len(scheduler) == 1 and scheduler[0][1:3] == ['COMPLETED', '0:0'],
            'loading-boundary allocation did not complete cleanly')
    return job


def verify(root, request_ref, observation_ref, cold_references, execution, cold_executions):
    """Strict cold/full-action comparison; never grants production release."""
    import os
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    from .training_sequential_augmentation import compare_physics_reference
    from .training_trajectory_audit import audit
    from ispatialgym.oracle.start_state import compare
    root = Path(root).resolve()
    q, sources, plan = inputs(root, checked(root, request_ref))
    observed = json.loads(checked(root, observation_ref).read_text())
    require(observed['request'] == request_ref, 'loading-boundary observation belongs to another request')
    job = executed_job(root, observed, execution)
    children = []
    for segment in observed['segments']:
        child = json.loads(checked(root, segment['probe']).read_text()); children.append(child)
        boundary = json.loads(checked(root, segment['boundary_observation']).read_text())
        require(boundary == {k:v for k,v in segment.items() if k != 'boundary_observation'}
                and str(boundary['scheduler_job_id']) == job, 'child boundary provenance differs')
    accounting = validate_sequence(q, observed, children)
    require(set(cold_references) == set(cold_executions) == {'0', '1'},
            'independent cold A and B references and execution proofs required')
    cold_entries = {}
    for key, reference in cold_references.items():
        cold = json.loads(checked(root, reference).read_text())
        cold_job = executed_job(root, cold, cold_executions[key])
        cold_q, _, _ = inputs(root, checked(root, cold['request']))
        require(cold_q['control'] == ('FRESH_A' if key == '0' else 'FRESH_B')
                and cold_q['sources'] == q['sources'] and cold_q['native_runtime'] == q['native_runtime'],
                'independent cold source or runtime differs')
        cold_children = [json.loads(checked(root, s['probe']).read_text()) for s in cold['segments']]
        validate_sequence(cold_q, cold, cold_children)
        require(reference != observation_ref and cold['scheduler_job_id'] != observed['scheduler_job_id'],
                'independent cold reference reused the tested allocation')
        cold_entries[key] = (cold_children[0]['entries'][0], cold_job)
    comparisons = []; sessions = []; report_paths = set()
    for spec, segment, child in zip(plan, observed['segments'], children):
        source = sources[spec['source']]['row']; cold_entry, cold_job = cold_entries[str(spec['source'])]
        cold_report = json.loads(checked(root, cold_entry['report']).read_text())
        cold_case = report_case(cold_report, cold_job)
        child_root = Path(segment['probe']['path']).resolve().parent
        require(Path(segment['boundary_observation']['path']).resolve().parent == child_root,
                'child boundary and native output directories differ')
        identities = []; child_sessions = set()
        for entry in child['entries']:
            report_path = checked(root, entry['report'])
            require(report_path.is_relative_to(child_root) and report_path not in report_paths,
                    'native report was copied or reused across loading episodes')
            report_paths.add(report_path); identities.append(entry['native_identity'])
            report = json.loads(checked(root, entry['report']).read_text()); case = report_case(report, job)
            child_sessions.add(case['boundaries'][0]['process_session'])
            require(case['boundaries'][0]['process_session'] != cold_case['boundaries'][0]['process_session'],
                    'cold and tested episode share a native process')
            require(report['instance_id'] == cold_report['instance_id'] == source['instance_id']
                    and report['replay'] == cold_report['replay'] == source['collector_demo']['path'],
                    'loading-boundary replay source differs')
            compare_physics_reference(cold_case, case)
            require(entry['after_episode_static_bodies'] == cold_entry['after_episode_static_bodies'],
                    'loading-boundary native initialized properties changed')
            require(entry['before_reuse_static_bodies'] is None or
                    entry['before_reuse_static_bodies'] == cold_entry['after_episode_static_bodies'],
                    'compatible reused world properties differ before restoration')
            start = compare(cold_report['start_state'], report['start_state'], tol={'vel':1e-4})
            require(start['ok'], 'loading-boundary native start differs from independent cold')
            trajectory = audit(root, cold_entry['report'], entry['report'])
            require(trajectory['status'] == 'matched', 'loading-boundary full native trajectory differs')
            comparisons.append({'source': spec['source'], 'report': entry['report'],
                                'start_state': start, 'trajectory': trajectory})
        require(len(child_sessions) == 1 and all(v == identities[0] for v in identities),
                'compatible child replaced its native world or actors')
        sessions.extend(child_sessions)
    require(len(set(sessions)) == len(plan), 'fresh-world segments reused a native process session')
    return {'schema':'isg-retained-loading-boundary-review-v1', 'status':'verified',
            'request':request_ref, 'observation':observation_ref, 'execution':execution,
            'independent_colds':cold_references, 'cold_executions':cold_executions,
            'comparisons':comparisons, **accounting,
            'scope':'Exact retained room loads/compatible AA, native reset and full actions. '
                    'Separate rendering/export and support-height coverage remain required.'}
