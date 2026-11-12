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


def ref(path): return {'path':str(path), 'sha256':file_digest(path)}


def checked(root, reference):
    path = Path(reference['path']).resolve()
    require(path.is_relative_to(root) and file_digest(path) == reference['sha256'],
            'mechanism source changed or escaped campaign')
    return path


def scope(request):
    require(request.get('schema') == 'isg-retained-mechanism-replay-request-v1'
            and request.get('campaign_id') == 'campaign_20261110'
            and request.get('control') in ORDERS and request.get('physical_delta') == DELTA
            and request.get('benchmark_credit') == 0 and request.get('production_released') is False
            and request.get('max_native_episodes') == len(ORDERS[request['control']])
            and request.get('timeout_seconds') == 1800,
            'unbounded mechanism replay request')
    return ORDERS[request['control']]


def inputs(root, request_path, *, runtime=None):
    from .production_gate import read_environment
    from .acceptance import report_passed
    root = Path(root).resolve(); path = Path(request_path).resolve()
    require(path.is_relative_to(root), 'mechanism request outside campaign')
    q = json.loads(path.read_text()); scope(q)
    manifest = json.loads(checked(root, q['source_manifest']).read_text())
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
            and proof['physical_delta'] == DELTA and proof['production_released'] is False,
            'mechanism candidate proof differs')
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
    types = {'A_FAILURE_A':ControlledEpisodeFailure,'A_INTERRUPT_A':ControlledEpisodeInterrupt}
    tro = source['source_tro']['path']; demo = source['collector_demo']['path']
    signature = runner.instance_signature(tro); directory = runner.resolve_instance(tro)[1]
    compatible, reason = runner.template_compatible(directory,tro,directory,tro)
    require(compatible, 'original native template cannot reuse itself: '+reason)
    shared = {'signature':signature,'compatible_template_dirs':{os.path.abspath(directory)}}
    result = {'schema':'isg-retained-mechanism-replay-observation-v1', 'request':ref(request_path),
        'source_manifest':q['source_manifest'], 'native_runtime':q['native_runtime'],
        'source_generation_runtime':source['generation_runtime'], 'source_collector_runtime':source['collector_runtime'],
        'physical_delta':DELTA, 'entries':[], 'benchmark_credit':0, 'production_released':False,
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
