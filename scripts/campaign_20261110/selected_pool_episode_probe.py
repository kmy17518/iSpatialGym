"""Observe exact pool episodes across production-compatible world boundaries.

No sampling, goal overrides, source rewriting, or compatibility bypass. The
same Python/Kit process clears its world when the production identities differ.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import time


def ref(path):
    path = Path(path)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def checked(reference):
    if ref(reference['path'])['sha256'] != reference['sha256']:
        raise ValueError('changed input: ' + reference['path'])
    return json.loads(Path(reference['path']).read_text())


def boundary(previous, current, compatible=False):
    if previous is None:
        return 'load'
    return 'reuse' if previous == current and compatible else 'reload'


def inputs(request_path):
    value = json.loads(Path(request_path).read_text())
    assert value['schema'] == 'isg-selected-pool-episode-probe-v1'
    assert value['benchmark_credit'] == 0 and value['production_released'] is False
    assert value['sequence'] == ['A', 'B', 'A'] and len(value['episodes']) == 3
    root = Path(value['campaign_root']).resolve()
    runtime = Path(value['native_runtime']['path']).parent
    assert runtime.is_relative_to(root/'code')
    assert ref(runtime/'.snapshot_commit') == value['native_runtime']
    records = []
    for episode in value['episodes']:
        review = checked(episode['review'])
        assert review['validation_passed'] and not review['accepted'] and review['quota_consumed'] == 0
        record = review['export_record']
        assert record['validation_key']['code_sha256'] == value['native_runtime']['sha256']
        assert record['instance_id'] == episode['instance_id']
        for artifact in record['artifacts'].values():
            assert Path(artifact['path']).resolve().is_relative_to(root)
            assert ref(artifact['path']) == artifact
        cfg = checked(episode['configuration'])
        assert cfg['validation_key'] == record['validation_key']
        assert ref(episode['source_environment']['path']) == episode['source_environment']
        assert episode['execution_environment']['ISPATIALGYM_CAMPAIGN_CONFIG'] == episode['configuration']['path']
        assert all(k.startswith(('ISPATIALGYM_', 'ISG_')) for k in episode['execution_environment'])
        scene = checked(record['artifacts']['recorded_scene'])
        assert scene['init_info']['args']['scene_model'] == cfg['template']['scene']
        records.append(record)
    assert value['episodes'][0]['expected']['target']['model'] == value['episodes'][2]['expected']['target']['model']
    assert value['episodes'][0]['expected']['target']['model'] != value['episodes'][1]['expected']['target']['model']
    return value, records


def live_identity(scene, expected):
    objects = {obj.name: obj for obj in scene.objects}
    result = {'objects': {}, 'pool_object_names': sorted(n for n in objects if n.startswith('campaign_pool_'))}
    assert result['pool_object_names'] == [expected['target']['native_name']], 'stale or missing pool object'
    for role in ('target', 'support'):
        spec = expected[role]; obj = objects[spec['native_name']]
        scale = [float(x) for x in obj.scale]
        assert obj.category == spec['category'] and obj.model == spec['model'], 'live role identity differs'
        assert scale == spec['scale'], 'live exact float32 scale differs'
        if role == 'support':
            assert expected['room'] in obj.in_rooms, 'wrong support room'
        result['objects'][role] = {'name': obj.name, 'category': obj.category, 'model': obj.model,
                                   'scale': scale, 'in_rooms': list(obj.in_rooms or [])}
    result['scene_expected'] = expected['scene']; result['task_room'] = expected['room']
    return result


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    temporary.replace(path)


def run(request_path, output, kind):
    value, records = inputs(request_path)
    output = Path(output).resolve(); root = Path(value['campaign_root']).resolve()
    assert output.is_relative_to(root) and not output.exists()
    output.mkdir(parents=True)
    os.environ['ISGF_ROOT'] = str(root)
    os.environ['ISPATIALGYM_FAST_COLLECTOR_PREBUILD'] = '1'
    import ispatialgym
    assert Path(ispatialgym.__file__).resolve().is_relative_to(Path(value['native_runtime']['path']).parent)
    import omnigibson as og
    result = {'schema': 'isg-selected-pool-episode-observation-v1', 'request': ref(request_path),
              'kind': kind, 'pid': os.getpid(), 'native_runtime': value['native_runtime'], 'episodes': [],
              'world_loads': 0, 'world_reloads': 0, 'world_reuses': 0,
              'benchmark_credit': 0, 'production_released': False, 'started_at': time.time()}
    write(output/'observation.json', result)
    previous_key = None; previous_tro = None; shared = {}
    previous_environment_keys = set()
    try:
        if kind == 'physics':
            from ispatialgym.oracle import run_isg as runner
            original_restore = runner.InstanceLoader.restore
        else:
            assert kind == 'render'
            from ispatialgym.multienv import replay as renderer
            original_start = renderer.Slot.start
        for i, (episode, record) in enumerate(zip(value['episodes'], records)):
            for name in previous_environment_keys:
                os.environ.pop(name, None)
            os.environ.update(episode['execution_environment'])
            previous_environment_keys = set(episode['execution_environment'])
            os.environ.pop('ISPATIALGYM_HEAD_CAMERA_ONLY', None)
            os.environ.pop('ISPATIALGYM_PHYSICS_ONLY', None)
            os.environ['ISPATIALGYM_FAST_COLLECTOR_PREBUILD'] = '1'
            os.environ['ISPATIALGYM_FAST_PARTIAL_SCENE'] = episode['replay_partial_scene']
            destination = output/f'episode_{i+1}'
            tro = record['artifacts']['tro']['path']
            compatible = False; reason = 'first world'
            if kind == 'physics':
                key = runner.instance_signature(tro)
                if previous_tro is not None and previous_key == key:
                    compatible, reason = runner.template_compatible(runner.resolve_instance(previous_tro)[1], previous_tro,
                                                                   runner.resolve_instance(tro)[1], tro)
                elif previous_tro is not None:
                    reason = 'native instance_signature differs; production queue would request a fresh world'
            else:
                key = record['render_compatibility_sha256']
                compatible = previous_key == key
                reason = 'exact render compatibility match' if compatible else 'render compatibility differs; separate exact manifest/world required'
            mode = boundary(previous_key, key, compatible)
            # Renderer main owns its world per invocation. Never claim reuse just
            # because a content key matches when this adapter does not retain it.
            if kind == 'render' and mode == 'reuse':
                mode = 'reload'; reason += '; adapter uses a separate native renderer invocation'
            clear = None
            if mode == 'reload':
                shared.clear(); gc.collect(); before = len(og.sim.scenes)
                og.clear(); after = len(og.sim.scenes)
                assert after == 0, 'native clear left a world loaded'
                clear = {'before_scene_count': before, 'after_scene_count': after, 'observed_at': time.time()}
                result['world_reloads'] += 1
            result['world_reuses' if mode == 'reuse' else 'world_loads'] += 1
            entry = {'position': i+1, 'instance_id': record['instance_id'], 'configuration': episode['configuration'],
                     'review': episode['review'], 'transition': mode, 'reason': reason, 'compatibility_key': key,
                     'clear': clear, 'started_at': time.time(), 'pid': os.getpid(), 'expected': episode['expected']}
            result['episodes'].append(entry); write(output/'observation.json', result)
            observations = []
            if kind == 'physics':
                def observed_restore(loader, *args, **kwargs):
                    answer = original_restore(loader, *args, **kwargs)
                    assert loader.cfg['scene']['scene_model'] == episode['expected']['scene']
                    observations.append(live_identity(loader.env.scene, episode['expected']))
                    return answer
                runner.InstanceLoader.restore = observed_restore
                argv = ['--instance', tro, '--mode', 'replay', '--replay', record['artifacts']['demo']['path'],
                        '--record', 'hdf5', '--no-goal-override', '--out', str(destination)]
                args = runner.parse_args(argv)
                if mode == 'reuse':
                    shared.setdefault('compatible_template_dirs', set()).add(runner.resolve_instance(tro)[1])
                rc, native_result = runner.run_instance(args, tro, shared)
                report = destination/record['instance_id']/'meta.json'
                entry.update(returncode=rc, native_result=native_result, report=ref(report), live_identity=observations)
                assert observations and rc == 0 and native_result.get('success') is True, 'native replay failed'
            else:
                def observed_start(slot, demo, *args, **kwargs):
                    answer = original_start(slot, demo, *args, **kwargs)
                    observations.append(live_identity(slot.scene, episode['expected']))
                    return answer
                renderer.Slot.start = observed_start
                destination.mkdir(); manifest = destination/'canary.jsonl'; manifest.write_text(json.dumps(record)+'\n')
                listing = destination/'demos.txt'; listing.write_text(str(Path(record['artifacts']['demo']['path']).parent)+'\n')
                results_path = destination/'results.jsonl'
                rc = renderer.main(['--demos', '@'+str(listing), '--campaign-manifest', str(manifest),
                     '--root', str(destination/'rendered'), '--task', record['variant'], '--slots', '1', '--scene', 'recorded',
                     '--results', str(results_path), '--repo-prefix', 'selected_pool_position_'+str(i+1)])
                rows = [json.loads(line) for line in results_path.read_text().splitlines()]
                rendered = [row for row in rows if row.get('tag') == 'REPLAY_DEMO']
                videos = [ref(p) for p in sorted((destination/'rendered').rglob('*.mp4'))]
                entry.update(returncode=rc, native_results=rendered, results=ref(results_path), videos=videos, live_identity=observations)
                assert observations and rc == 0 and len(rendered) == 1 and rendered[0]['kept'] and videos, 'native rendering incomplete'
                assert rendered[0]['campaign_restoration']['scale_verified']
            entry['finished_at'] = time.time(); previous_key = key; previous_tro = tro
            write(output/'observation.json', result)
        inputs(request_path)
        result.update(status='passed_exact_three_episode_sequence', finished_at=time.time())
        write(output/'observation.json', result)
    except BaseException as error:
        import traceback
        result.update(status='failed', error=repr(error), traceback=traceback.format_exc(), finished_at=time.time())
        write(output/'observation.json', result)
        raise
    finally:
        if kind == 'physics': runner.InstanceLoader.restore = original_restore
        else: renderer.Slot.start = original_start
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--kind', required=True, choices=['physics', 'render'])
    arguments = parser.parse_args()
    try:
        print(json.dumps(run(arguments.request, arguments.output, arguments.kind)), flush=True)
    finally:
        import sys
        og = sys.modules.get('omnigibson')
        if og is not None: og.shutdown()
