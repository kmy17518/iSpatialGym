"""Bounded ordinary-loader replay of exact successful canonical mechanism sources.

Original generation identities remain intact. This driver neither samples nor
accepts data; each candidate cold reference must be verified independently.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

from .contracts import file_digest, require, write_json_atomic

ORDERS = {'FRESH_A':[0], 'AA':[0,0], 'A_FAILURE_A':[0,0], 'A_INTERRUPT_A':[0,0]}
DELTA = {'source_inputs_actions_changed':False, 'native_loader_changed_by_probe':False,
         'additional_physics_steps':0, 'additional_stop_play_pairs':0,
         'explicit_sdk_reconstructions':0, 'scope':'ordinary shared loader, exact repeated source only',
         'changed_models_or_augmentation_tested':False}
REBUILD_DELTA = {**DELTA, 'native_loader_changed_by_probe':True,
    'additional_stop_play_pairs':1, 'explicit_sdk_reconstructions':1,
    'counter_scope':'per episode, cold and reused',
    'scope':'shared late incoming-state reconstruction, exact repeated source only',
    'boundary':'incoming_state_loaded_immediately_before_existing_first_reset_physics_step',
    'reference_state_copied':False, 'static_properties_required_bit_identical':True,
    'thresholds_changed':False}


def physical_delta(request):
    mode = request.get('physics_reconstruction')
    require(mode in (None, 'late_incoming_v1'), 'unknown native reconstruction')
    value = dict(REBUILD_DELTA if mode else DELTA)
    full = request.get('complete_native_snapshots')
    require(full in (None, True), 'unknown native snapshot scope')
    if full:
        value['complete_native_snapshots'] = 'existing loader update_initial_file calls include every initialized actor; prior recording filter restored'
    return value


def install_complete_snapshots(runner):
    from ispatialgym.oracle.campaign_restore import native_snapshot_scope
    original = runner.InstanceLoader.restore
    def restored(loader, *args, **kwargs):
        scene = loader.env.scene; native = scene.update_initial_file
        had = 'update_initial_file' in vars(scene); prior = vars(scene).get('update_initial_file')
        events = []; loader.report['complete_native_snapshot_inventory'] = events
        def updated(*args, **kwargs):
            result = native(*args, **kwargs)
            names = set(scene._initial_file['state']['registry']['object_registry'])
            expected = {obj.name for obj in scene.objects}
            events.append({'snapshot_objects':sorted(names), 'native_objects':sorted(expected),
                           'complete':names == expected, 'physics_steps_added':0})
            require(names == expected, 'native loader snapshot omitted initialized actors')
            return result
        with native_snapshot_scope(scene, loader.report):
            scene.update_initial_file = updated
            try:
                result = original(loader, *args, **kwargs)
                require(events, 'native restore did not publish a snapshot')
                return result
            finally:
                if had: scene.update_initial_file = prior
                else: del scene.update_initial_file
    runner.InstanceLoader.restore = restored


def ref(path): return {'path':str(path), 'sha256':file_digest(path)}


def checked(root, reference):
    path = Path(reference['path']).resolve()
    require(path.is_relative_to(root) and file_digest(path) == reference['sha256'],
            'mechanism source changed or escaped campaign')
    return path


def scope(request):
    require(request.get('schema') == 'isg-retained-mechanism-replay-request-v1'
            and request.get('campaign_id') == 'campaign_20261110'
            and request.get('control') in ORDERS and request.get('physical_delta') == physical_delta(request)
            and request.get('benchmark_credit') == 0 and request.get('production_released') is False
            and request.get('max_native_episodes') == len(ORDERS[request['control']])
            and request.get('timeout_seconds') == 1800,
            'unbounded mechanism replay request')
    return ORDERS[request['control']]


def additional_source(root, manifest, request):
    """Consume the actual camera/block review schema without fabricating a group."""
    classes = {'camera':'camera_relocalization-standard',
               'grouped_objects':'configuration_matching-block_design_01'}
    require(request['mechanism'] in classes, 'additional source mechanism unsupported')
    rows = [r for r in manifest['rows'] if r['class_id'] == classes[request['mechanism']]
            and r['instance_id'] == request['instance_id']]
    require(len(rows) == 1, 'additional mechanism source ambiguous')
    row = dict(rows[0])
    for name in ('source_tro','source_sidecar','collector_environment','collector_meta','collector_demo',
                 'own_replay_gate','own_replay_meta','generation_row','campaign_config','first_party_strict_review'):
        checked(root, row[name])
    original = row['source_generation_and_collection_runtime']
    marker = Path(original['path'])/'.snapshot_commit'
    checked(root, {'path':str(marker),'sha256':original['snapshot_sha256']})
    require(marker.read_text().strip() == original['commit'], 'additional original runtime changed')
    checked(root, original['b1k'])
    identity = {**original, 'identity':ref(marker)}
    row.update(generation_runtime=identity, collector_runtime=identity,
               mechanism=request['mechanism'])
    gate = json.loads(Path(row['own_replay_gate']['path']).read_text())
    require(gate.get('success') is True and gate.get('replay_rc') == 0
            and gate.get('demo') == row['collector_demo']['path']
            and gate.get('demo_sha256') == row['collector_demo']['sha256']
            and gate.get('instance') == row['source_tro']['path']
            and gate.get('tro_sha256') == row['source_tro']['sha256']
            and row['own_replay_meta']['path'] in gate.get('replay_meta_paths',[]),
            'additional source lacks its exact own replay')
    review = json.loads(Path(row['first_party_strict_review']['path']).read_text())
    schema = 'isg-camera-release-review-v1' if request['mechanism'] == 'camera' else 'isg-release-review-v1'
    require(review.get('schema') == schema and review.get('class_id') == row['class_id']
            and review.get('template_validation_passed') is True
            and review.get('gate') == {k:row['own_replay_gate'][k] for k in ('path','sha256')},
            'additional source lacks its first-party strict review')
    members = [m for m in review.get('members',[]) if m.get('instance_id') == row['instance_id']]
    require(len(members) == 1, 'additional reviewed source missing or ambiguous')
    # These are zero-credit retained-source controls. Camera grouping is exempt;
    # the actual noncamera source review remains intact and is not reinterpreted.
    if request['mechanism'] == 'grouped_objects':
        require(row['instance_id'] in review.get('selected_member_ids',[])
                and all(members[0].get(key+'_sha256') == row[field]['sha256'] for key, field in (
                    ('demo','collector_demo'),('tro','source_tro'),('report','collector_meta'),
                    ('sidecar','source_sidecar'))), 'grouped source review artifacts differ')
    return row


def inputs(root, request_path, *, runtime=None):
    from .production_gate import read_environment
    from .acceptance import report_passed
    root = Path(root).resolve(); path = Path(request_path).resolve()
    require(path.is_relative_to(root), 'mechanism request outside campaign')
    q = json.loads(path.read_text()); scope(q)
    manifest = json.loads(checked(root, q['source_manifest']).read_text())
    if manifest['schema'] == 'isg-pipeline-additional-source-bindings-v1':
        row = additional_source(root, manifest, q)
    else:
        require(manifest['schema'] == 'isg-pipeline-canonical-source-bindings-v1', 'unknown source binding schema')
        rows = [r for r in manifest['rows'] if r['mechanism'] == q['mechanism']]
        require(len(rows) == 1 and rows[0]['instance_id'] == q['instance_id'], 'mechanism source ambiguous')
        row = rows[0]
        for name in ('native_request','native_review','native_configuration','goal_plan','native_environment',
                     'source_tro','source_sidecar','collector_environment','collector_meta','collector_demo',
                     'own_replay_gate','own_replay_meta','strict_distinct_group_review'):
            checked(root, row[name])
        for name in ('generation_runtime','collector_runtime','own_replay_runtime'):
            marker = checked(root, row[name]['identity'])
            require(marker.read_text().strip() == row[name]['commit'], 'original source runtime differs')
        review = row['strict_review_entry']
        require(review['eligible_distinct_group'] is True and any(
            member['index'] == row['index'] and member['native_quality']['passed'] is True
            for member in review['successful_members']), 'source lacks its own strict distinct-group evidence')
    for name, mode in (('collector_meta','oracle'),('own_replay_meta','replay')):
        report = json.loads(Path(row[name]['path']).read_text()); report_passed(report, mode)
        require(report['instance_id'] == row['instance_id'], 'source report instance differs')
    source = json.loads(Path(row['source_tro']['path']).read_text())['ispatialgym']
    require(source['instance_id'] == row['instance_id'] and source['campaign']['class_id'] == row['class_id']
            and source['campaign'].get('runtime',{}).get('purpose') != 'training_production',
            'mechanism source requires a separate production-purpose execution contract')
    environment = {k:v for k,v in read_environment(row['collector_environment']['path']).items()
                   if k.startswith(('ISPATIALGYM_','ISG_'))}
    require(q['source_environment'] == environment, 'mechanism execution source environment changed')
    runtime = Path(runtime).resolve() if runtime else Path(__file__).resolve().parents[2]
    require(ref(runtime/'.snapshot_commit') == q['native_runtime'], 'mechanism candidate runtime differs')
    proof = json.loads(checked(root, q['runtime_proof']).read_text())
    require(proof['candidate_runtime'] == q['native_runtime'] and proof['source_manifest'] == q['source_manifest']
            and proof['physical_delta'] == physical_delta(q) and proof['production_released'] is False,
            'mechanism candidate proof differs')
    if q.get('physics_reconstruction'):
        for name in ('installed_api_bindings','installed_api_definition','installed_simulation_route_test'):
            item = proof[name]
            require(Path(item['path']).is_absolute() and file_digest(item['path']) == item['sha256'],
                    'inspected native reconstruction API changed')
    actual = {str(p.relative_to(runtime)):file_digest(p) for folder in ('ispatialgym','scripts')
              for p in (runtime/folder).rglob('*.py') if p.is_file()}
    require(actual == proof['candidate_python_payloads'], 'integrated candidate payload changed')
    return q, row, environment


def run(root, request_path, output, *, preflight=False):
    root = Path(root).resolve(); output = Path(output).resolve()
    require(output.is_relative_to(root) and not output.exists(), 'private mechanism output required')
    q, source, environment = inputs(root, request_path)
    if preflight: return {'passed':True,'native_execution':False,'request':ref(request_path),
                         'source_manifest':q['source_manifest'],'candidate_runtime':q['native_runtime']}
    output.mkdir(parents=True)
    os.environ.update(environment)
    os.environ.update(ISGF_ROOT=str(root), ISPATIALGYM_FAST_COLLECTOR_PREBUILD='1',
        ISPATIALGYM_CAMPAIGN_RESTORATION_PROBE='1', ISPATIALGYM_CAMPAIGN_RESTORATION_CONTEXT='1')
    from ispatialgym.oracle import run_isg as runner
    from .training_batch_probe import (install_reset_boundary_probe, ControlledEpisodeFailure,
                                      ControlledEpisodeInterrupt)
    from .training_pool_probe import world_construction_counter
    from .training_late_rebuild_probe import static_bodies
    from .acceptance import report_passed
    import omnigibson as og
    require(Path(runner.__file__).resolve().is_relative_to(Path(q['native_runtime']['path']).parent),
            'native mechanism loader imported outside candidate')
    cleanups = []; abort = {'enabled':False}
    install_reset_boundary_probe(runner, final_pre_action=True, next_native_control_step=True,
                                 pending_cleanups=cleanups, abort_control=abort)
    if q.get('physics_reconstruction'):
        from .training_late_rebuild_probe import install
        from ispatialgym.oracle.campaign_restore import restart_loaded_physics
        def restart(meta, sim, *, execution=None, while_stopped=None):
            require(meta.get('campaign_id') == root.name and meta.get('class_id') == source['class_id']
                    and meta.get('runtime',{}).get('purpose') != 'training_production',
                    'reconstruction source scope changed')
            event = restart_loaded_physics(sim, while_stopped=while_stopped)
            event['boundary'] = REBUILD_DELTA['boundary']
            return event
        install(runner, {'late_incoming_physics_rebuild_v1':True,
                         'runtime_compatibility':q['runtime_proof']}, output, restart=restart)
    if q.get('complete_native_snapshots'):
        install_complete_snapshots(runner)
    types = {'A_FAILURE_A':ControlledEpisodeFailure,'A_INTERRUPT_A':ControlledEpisodeInterrupt}
    tro = source['source_tro']['path']; demo = source['collector_demo']['path']
    signature = runner.instance_signature(tro); directory = runner.resolve_instance(tro)[1]
    compatible, reason = runner.template_compatible(directory,tro,directory,tro)
    require(compatible, 'original native template cannot reuse itself: '+reason)
    shared = {'signature':signature,'compatible_template_dirs':{os.path.abspath(directory)}}
    result = {'schema':'isg-retained-mechanism-replay-observation-v1', 'request':ref(request_path),
        'source_manifest':q['source_manifest'], 'native_runtime':q['native_runtime'],
        'source_generation_runtime':source['generation_runtime'], 'source_collector_runtime':source['collector_runtime'],
        'physical_delta':physical_delta(q), 'entries':[], 'benchmark_credit':0, 'production_released':False,
        'started_at':time.time(), 'world_compatibility':{'signature':signature,'native_check':reason}}
    try:
        with world_construction_counter(og) as world:
            result['world_construction'] = world; identity = None
            for ordinal in range(len(scope(q))):
                abort.update(enabled=q['control'] in types and ordinal == 0,
                             exception_type=types.get(q['control'],ControlledEpisodeFailure))
                destination = output/('episode_'+str(ordinal))
                argv = ['--instance',tro,'--mode','replay','--replay',demo,'--record','hdf5',
                        '--no-goal-override','--out',str(destination)]
                before = static_bodies(shared['env'].scene) if shared.get('env') else None
                try: code, entry = runner.run_instance(runner.parse_args(argv),tro,shared)
                finally:
                    while cleanups: cleanups.pop()()
                report_path = destination/source['instance_id']/'meta.json'
                report = json.loads(report_path.read_text()); scene = shared['env'].scene
                current = {'environment':id(shared['env']),'scene':id(scene),
                           'actors':{o.name:id(o)for o in scene.objects}}
                require(identity is None or identity == current, 'native mechanism world/actors replaced')
                identity = current
                record = {'ordinal':ordinal,'returncode':code,'result':entry,'report':ref(report_path),
                          'native_identity':current,'before_reuse_static_bodies':before,
                          'after_episode_static_bodies':static_bodies(scene),'expected_abort':abort['enabled']}
                result['entries'].append(record); write_json_atomic(output/'probe.json',result)
                if abort['enabled']:
                    proof = report.get('loader',{}).get('campaign_restoration',{}).get('controlled_abort',{})
                    require(code != 0 and entry.get('success') is False
                            and entry.get('error','').startswith(types[q['control']].__name__+'(')
                            and proof.get('injected') is True
                            and proof.get('exception_type') == types[q['control']].__name__
                            and proof.get('native_control_calls_completed') == 1,
                            'native mechanism did not exercise the exact one-action cleanup')
                else:
                    require(code == 0 and entry.get('success') is True,'native mechanism action replay failed')
                    report_passed(report,'replay')
            require(world == {'attempted':1,'completed':1}, 'mechanism replay rebuilt its world')
        inputs(root,request_path)
        result.update(status='native_mechanism_controls_completed',finished_at=time.time())
    except BaseException as error:
        result.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        write_json_atomic(output/'failure.json',{'request':ref(request_path),'error':repr(error),
                                               'traceback':result['traceback']})
        raise
    finally: write_json_atomic(output/'probe.json',result)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--request',required=True)
    p.add_argument('--output',required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args()
    try: print(json.dumps(run(a.root,a.request,a.output,preflight=a.preflight)),flush=True)
    finally:
        native=sys.modules.get('omnigibson')
        if native is not None:native.shutdown()
