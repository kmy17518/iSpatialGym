"""Strict, read-only audit of changed-augmentation one-world restoration.

Independent references, ordinary success, ordinary failure and interruption are
separate witnesses. This audit neither dispatches nor releases production.
"""
import math
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


def observed_augmentation_change(root, records, first, second):
    """Prove real light/parking changes, even if cold and reused paths share a bug."""
    require(len(records) == 2, 'two augmentation sources required')
    def vector(value, length):
        require(isinstance(value, (list, tuple)) and len(value) == length and
                all(type(x) in (int, float) and math.isfinite(x) for x in value),
                'invalid native augmentation vector')
        return value
    def near(actual, expected):
        return (type(actual) in (int, float) and type(expected) in (int, float) and
                math.isfinite(actual) and math.isfinite(expected) and
                abs(actual-expected) <= 1e-5*max(1., abs(expected)))
    stable_refs=[]; native=[]
    for row in records:
        setup=row['template']['scene_setup']
        stable_refs.append(setup.get('sequential_augmentation_base', {}).get('scene') or
                           setup['training_height_dependency']['stable_scene'])
        tro=checked(root,row['artifacts']['tro'])['ispatialgym']['campaign']
        require(tro['perturbations'] == row['perturbations'], 'native augmentation source differs')
        native.append(tro['runtime']['augmentation'])
    require(stable_refs[0] == stable_refs[1], 'augmentation stable references differ')
    stable=checked(root,stable_refs[0]); states=stable['state']['registry']['object_registry']
    parked=[set(row['perturbations']['removed_objects']) for row in records]
    changed=parked[0] ^ parked[1]
    require(changed, 'augmentation has no declared parking transition')
    require(records[0]['perturbations']['lighting'] != records[1]['perturbations']['lighting'],
            'augmentation has no declared lighting transition')
    for row, source, observation, removed in zip(records,native,(first,second),parked):
        light=source['lighting']
        require(light['request'] == row['perturbations']['lighting'], 'native lighting request differs')
        require(set(observation['lights']) == set(light['fixtures']), 'native lighting fixture inventory differs')
        require(all(near(observation['lights'][name],value['applied']) for name,value in light['fixtures'].items()),
                'observed fixture lighting does not match declared augmentation')
        sky=light['skybox']['applied']; actual=observation['skybox']
        require(near(actual['intensity'],sky['intensity']) and
                all(near(a,b) for a,b in zip(vector(actual['color'],3),vector(sky['color'],3))),
                'observed skybox does not match declared augmentation')
        for name in parked[0] | parked[1]:
            info=stable['objects_info']['init_info'][name]['args']
            require(info.get('fixed_base') is True and name in observation['objects'] and
                    observation['objects'][name]['fixed_base'] is True,
                    'declared parking actor missing or not fixed in native observation')
            obj=observation['objects'][name]; baseline=states[name]['root_link']
            expected=source['moved_objects'][name]['moved_to'] if name in removed else baseline['pos']
            require(math.dist(vector(obj['position'],3),vector(expected,3)) <= .002,
                    'observed parking position does not match declared augmentation: '+name)
            qa,qb=vector(obj['orientation'],4),vector(baseline['ori'],4)
            norm=math.sqrt(sum(x*x for x in qa)*sum(x*x for x in qb))
            require(norm > 0 and 2*math.acos(min(1.,abs(sum(x*y for x,y in zip(qa,qb)))/norm)) <= 1e-4,
                    'observed parking orientation differs from stable scene: '+name)
    require(first['lights'].keys() == second['lights'].keys(), 'native lighting inventories differ')
    light_changed=[name for name in first['lights'] if not near(first['lights'][name],second['lights'][name])]
    sky_changed=(not near(first['skybox']['intensity'],second['skybox']['intensity']) or
                 any(not near(a,b) for a,b in zip(first['skybox']['color'],second['skybox']['color'])))
    require(light_changed or sky_changed, 'declared lighting changed but native lighting did not')
    distances={name:math.dist(first['objects'][name]['position'],second['objects'][name]['position'])
               for name in sorted(changed)}
    require(all(distance > .004 for distance in distances.values()),
            'declared parking changed but native actor position did not')
    return {'status':'verified_observed_changes','changed_fixtures':light_changed,
            'skybox_changed':sky_changed,'parking_displacement_m':distances,
            'native_source_positions_and_lighting_checked':True,
            'independent_reference_state_sha256':[digest(first),digest(second)]}


def verify(root, reference, started_at):
    return _verify_stages(root,reference,started_at,(('physics',PHYSICS),('rendering',RENDER)))


def verify_rendering(root,reference,started_at):
    """Audit the complete rendering stage without claiming physics clearance."""
    return _verify_stages(root,reference,started_at,(('rendering',RENDER),))


def _verify_stages(root, reference, started_at, stages):
    document = checked(root, reference)
    full=len(stages)==2
    require(document.get('schema') == ('isg-sequential-augmentation-verification-input-v1'if full else
                'isg-sequential-rendering-verification-input-v1') and
            set(document['execution']) == {stage for stage,_ in stages} and
            all(set(document.get(stage,{}))==set(specs)and set(document['execution'][stage])==set(specs)
                for stage,specs in stages),
            'complete independent-reference/success/failure/interruption matrix required')
    sources = {}; source_refs = None; runtimes = {}; artifacts = set(); results = {}; all_processes = set()
    augmentation_changes = {}
    for stage, specifications in stages:
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
            identity = probe['probe_runtime']; runtimes.setdefault(stage,identity)
            require(identity == runtimes[stage] and file_digest(identity['path']) == request['probe_runtime_sha256'] == identity['sha256'],
                    'sequential matrix stage runtime differs')
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
        def loaded(case):
            entry=cases[case][0]
            return (entry['boundaries'][-1] if stage == 'physics' else entry['observation'])['loaded_state']
        augmentation_changes[stage]=observed_augmentation_change(root,
            [sources[digest(ref)]['export_record'] for ref in source_refs],loaded('fresh_A'),loaded('fresh_B'))
        results[stage] = stage_results
    return {'schema': 'isg-sequential-augmentation-verification-v1'if full else'isg-sequential-rendering-verification-v1', 'input': reference,
            'status': 'verified', 'runtimes': runtimes, 'sources': source_refs, 'stages': results,
            'verified_stages':[stage for stage,_ in stages],'full_matrix_verified':full,
            'independent_processes': len(all_processes), 'observed_augmentation_changes':augmentation_changes,
            'checks': ['changed lighting and fixed-actor parking',
                'one actual Environment construction per process', 'exact native task/model/scale/room/assets',
                'independent A and B references', 'A/B/A restoration', 'exact rendered artifacts']+
                (['successful A/A cleanup','failed A/A cleanup','interrupted A/A cleanup',
                  'unchanged reset/first-action tolerances']if full else[]),
            'production_released': False, 'benchmark_credit': 0}


def verify_render_cleanup(root,reference,started_at):
    """Audit native writer and post-render abort boundaries against fresh A."""
    document=checked(root,reference)
    require(document.get('schema')=='isg-sequential-render-cleanup-input-v1'and
            set(document['controls'])=={'failure','interruption'},'renderer cleanup matrix incomplete')
    original=checked(root,document['independent_A']['probe'])
    verify_probe_execution(root,document['independent_A']['probe'],document['independent_A']['execution'])
    request,records=inputs(root,original['request']['path'])
    require(request.get('diagnostic_control')=='FRESH_A'and len(original['entries'])==1 and
            original['kind']=='kinematic_observation_rendering'and
            original['entries'][0]['canary_review']==request['canaries'][0],
            'independent fresh A rendering required')
    require(file_digest(original['probe_runtime']['path'])==original['probe_runtime']['sha256']==request['probe_runtime_sha256']and
            verify_runtime_compatibility(root,request,records,Path(original['probe_runtime']['path']).parent)==original['runtime_compatibility'],
            'independent render runtime differs')
    first=checked(root,original['entries'][0]['report']);require(first.get('tag')=='REPLAY_DEMO'and first.get('kept')is True,'independent reference failed')
    baseline=first['campaign_restoration']['native_observation'];processes={one_world(original,[baseline])}
    source=validated_canary(root,original['entries'][0]['canary_review'],started_at);artifacts=set();results={}
    require(Path(first['demo']).name==source['instance_id']and first['campaign_restoration']['scale_verified']is True and
            baseline['validation_key']==source['export_record']['validation_key']and
            baseline['perturbations_sha256']==digest(source['export_record']['perturbations'])and
            baseline['loaded_state_sha256']==digest(baseline['loaded_state']),
            'independent render state binding differs')
    for name,expected,boundary in [('failure','ControlledRenderFailure','after_native_writer_action'),
                                   ('interruption','ControlledRenderInterrupt','after_existing_render')]:
        item=document['controls'][name];probe=checked(root,item['probe'])
        execution=verify_probe_execution(root,item['probe'],item['execution']);value,actual=inputs(root,probe['request']['path'])
        require(value['canaries']==request['canaries']and value.get('render_cleanup_only')is True and
                value['diagnostic_control']==('RENDER_FAILURE_A'if name=='failure'else'RENDER_INTERRUPT_A')and
                probe['kind']=='kinematic_observation_rendering'and len(probe['entries'])==2,'renderer cleanup exact source/order differs')
        require(probe['probe_runtime']==original['probe_runtime'],'normal and cleanup rendering revisions differ')
        require(file_digest(probe['probe_runtime']['path'])==probe['probe_runtime']['sha256']==value['probe_runtime_sha256'],
                'renderer cleanup executed runtime differs')
        require(verify_runtime_compatibility(root,value,actual,Path(probe['probe_runtime']['path']).parent)==probe['runtime_compatibility'],'renderer cleanup runtime differs')
        observations=[]
        for index,entry in enumerate(probe['entries']):
            require(entry['canary_review']==request['canaries'][0],'renderer cleanup did not return to exact A')
            report=checked(root,entry['report']);restoration=report['campaign_restoration'];observation=restoration['native_observation']
            require(Path(report['demo']).name==source['instance_id']and restoration['scale_verified']is True and
                    observation.get('scheduler_job_id')and observation.get('simulator_module','').startswith('omnigibson.')and
                    observation['validation_key']==source['export_record']['validation_key']and
                    observation['perturbations_sha256']==digest(source['export_record']['perturbations'])and
                    observation['loaded_state_sha256']==digest(observation['loaded_state']),'renderer cleanup native state identity differs')
            same_loaded_state(baseline['loaded_state'],observation['loaded_state']);observations.append(observation)
            if index==0:
                cleanup=report['cleanup'];require(entry.get('expected_controlled_failure')is True and
                    report['tag']=='REPLAY_EPISODE_ABORTED'and report['kept']is False and
                    cleanup['exception_type']==expected and cleanup['boundary']==boundary and
                    cleanup['frames_observed']==(2 if name=='failure'else 1)and
                    cleanup['native_writer_abort_called']is True and cleanup['trajectory_buffer_empty']is True and
                    cleanup['dataset_pending_frames']is False and cleanup['writer_aborted']is True and
                    cleanup['physics_steps_added']==cleanup['render_ticks_added']==0 and
                    cleanup['expected_controlled_failure']is True,'native renderer failure/interruption cleanup incomplete')
            else:
                require(report['tag']=='REPLAY_DEMO'and report['kept']is True and
                        isinstance(report['episode_index'],int)and entry.get('outputs'),'post-cleanup A render missing')
                for output in entry['outputs']:
                    key=(output['path'],output['sha256'])
                    if key in artifacts:continue
                    p=Path(output['path']).resolve();require(p.is_relative_to(Path(root).resolve())and p.stat().st_size>0 and file_digest(p)==output['sha256'],'post-cleanup renderer output changed');artifacts.add(key)
        process=one_world(probe,observations);require(process not in processes,'renderer cleanup control was not independent');processes.add(process)
        results[name]={'probe':item['probe'],'execution':execution,'world_load_count':1,'process_session':process,'boundary':boundary,'successful_following_A_report':probe['entries'][1]['report']}
    return {'schema':'isg-sequential-render-cleanup-verification-v1','status':'verified','input':reference,
            'independent_A':document['independent_A'],'controls':results,'benchmark_credit':0,'production_released':False,
            'scope':'Actual native writer-action failure and post-existing-render interruption; all partial native writer buffers cleared, following A in same world equals independent fresh A. Unexpected-error propagation remains covered by exact CPU guard regressions.'}
