"""Verify finite native query outputs without granting demo/replay/release credit."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main(args):
    execution, out, runtime = map(Path, (args.execution, args.output, args.runtime))
    sys.path[:0] = [str(runtime / 'Behavior-1K/bddl3'), str(runtime)]
    from ispatialgym.sampling import grasp_library as GL
    from ispatialgym.grasping import schema as GS
    hashes = {}
    def sha(p):
        p = str(p)
        if p not in hashes:
            hashes[p] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        return hashes[p]
    def ref(p):
        return {'path': str(p), 'sha256': sha(p)}
    def read(p):
        return json.loads(Path(p).read_text())
    assert not out.exists()
    out.mkdir(parents=True)
    rows = []
    for terminal in sorted(execution.glob('floating_*/terminal.json')):
        directory = terminal.parent
        row = {'terminal': ref(terminal), 'id': directory.name, 'benchmark_credit': 0}
        try:
            t = read(terminal)
            p = read(directory / 'prepared.json')
            assert t['prepared_sha256'] == sha(directory / 'prepared.json')
            assert t['physical_dedup_key'] == p['physical_dedup_key'] and t['configuration_ids'] == p['configuration_ids']
            assert p['native_runtime'] == str(runtime) and t['native_runtime'] == str(runtime)
            for r in p['input_refs']:
                assert sha(r['path']) == r['sha256'], r['path']
            row.update(physical_dedup_key=p['physical_dedup_key'], configuration_ids=p['configuration_ids'], class_id=p['class_id'], arm=p['arm'], target=p['target']['model_key'], job_id=t['job_id'], duration_seconds=t['finished_at'] - t['started_at'], prepared=ref(directory / 'prepared.json'))
            native = directory / 'native'
            summary_path = native / 'summary.json'
            summary = read(summary_path)
            errors = [v.get('error') for values in summary.get('problems', {}).values() for v in values.values() if v.get('error')]
            assert t['returncode'] == 0
            tros = list(native.rglob('*-tro_state.json'))
            assert len(tros) == t['tro_count']
            if not tros:
                row.update(status='verified_setup_routing_failure' if any('missing_grasps:floating' in x for x in errors) else 'verified_bounded_no_native_success', errors=errors, native_summary=ref(summary_path), physical_zero=False)
                rows.append(row)
                continue
            assert len(tros) == 1 and summary['n_written'] == 1
            tro_path = tros[0]
            assert ref(tro_path) == t['tros'][0]
            tro = read(tro_path)['ispatialgym']
            meta_path = tro_path.parent.parent / 'template_meta.json'
            meta = read(meta_path)
            assert tro['task_variant'] == p['class_id']
            assert tro['scene']['model'] == p['scene'] and tro['scene']['task_rooms'] == [p['room']]
            assert meta['choice']['rooms'] == [p['room']] and meta['choice']['scene'] == p['scene']
            target = p['target']
            choice = meta['choice']['models']['target']
            assert all(choice[k] == target[k] for k in ('category', 'model', 'dataset'))
            assert GS.canonical_scale(choice['scale']) == GS.canonical_scale(target['applied_scale'])
            template_path = Path(meta['template_path'])
            assert sha(template_path) == meta['template_sha256'] == tro['scene']['template_sha256']
            template = read(template_path)['objects_info']['init_info']
            binding = next(b for b in meta['bindings'] if b['role'] == 'target')
            actor = template[binding['scene_object_name']]['args']
            assert all(actor[k] == target[k] for k in ('category', 'model'))
            assert actor['dataset_name'] == target['dataset'] and all(abs(a-b) < 1e-6 for a, b in zip(actor['scale'], target['applied_scale']))
            floor = next(b for b in meta['bindings'] if b['role'] == 'floor')
            expected_floor = p['expected_native_floor']
            assert floor['scene_object_name'] == expected_floor['name'] and floor['model'] == expected_floor['args']['model']
            assert all(abs(a-b) < 1e-6 for a, b in zip(template[floor['scene_object_name']]['args']['scale'], expected_floor['args']['scale']))
            custom_path = Path(p['sample_argv'][p['sample_argv'].index('--customs') + 1])
            assert sha(custom_path) == tro['task_spec_sha256']
            bddls = list(Path(p['sample_argv'][p['sample_argv'].index('--bddl-root') + 1]).rglob('*.bddl'))
            assert len(bddls) == 1 and sha(bddls[0]) == tro['bddl_sha256'] == meta['bddl_sha256']
            os.environ.update(p['environment'])
            key = GL.resolve_key(target['category'], target['model'], choice.get('grasp_key'), ['floating'])
            lib = GL.v2_entry(key)
            assert lib is not None and lib.floating and lib.data['arm'] == p['arm']
            assert lib.scale == GS.canonical_scale(target['applied_scale'])
            assert all(lib.data[k] == target[k] for k in ('category', 'model', 'dataset'))
            grasps = {g.name: g for g in GL.load_grasps(key)}
            assert len(tro['grasps']) == 1
            grasp = tro['grasps'][0]
            assert grasp['arm'] == p['arm'] and grasp['grasp_key'] == key and grasp['grasp_name'] in grasps
            assert meta['hold_arm']['offered'] == [p['arm']] and meta['hold_arm']['arm'] == p['arm']
            instance_summary_path = tro_path.with_name(tro_path.name.replace('-tro_state.json', '.summary.json'))
            instance_summary = read(instance_summary_path)
            assert instance_summary['arm']['arm'] == p['arm'] and instance_summary['bddl_at_init']['init_conditions_all_true']
            check = instance_summary['program_check']
            assert check['mode'] == 'expand' and check['stats']['rejected'] is False
            assert check['stats']['n_actions'] == check['stats']['n_actions_checked'] == len(check['actions']) > 0
            assert all(action.get('witnesses') and p['arm'] in action['arms'] for action in check['actions'])
            sidecar_path = tro_path.parent / instance_summary['sidecar']
            sidecar = read(sidecar_path)
            assert sidecar['tro_sha256'] == sha(tro_path) and sidecar['waypoint_policy'] == 'checked'
            assert instance_summary['goal_occlusion']['occluded'] is False
            row.update(status='verified_native_configuration_query', tro=ref(tro_path), template=ref(template_path), metadata=ref(meta_path), native_summary=ref(summary_path), instance_summary=ref(instance_summary_path), sidecar=ref(sidecar_path), native_library=ref(lib.path), consumer_key=key, actions_checked=check['stats']['n_actions_checked'], actual_scale=actor['scale'], scope='Exact native binding, scale, arm, floor, source/runtime/input closure, positive v2 grasp and full native program expansion checked; demo, action replay and acceptance remain unproven.')
        except Exception as exc:
            row.update(status='reader_error_requires_diagnosis', error=repr(exc))
        rows.append(row)
    path = out / 'audit.json'
    report = {'schema': 'isg-native-configuration-query-audit-v1', 'observed_at': time.time(), 'runtime': ref(runtime / '.snapshot_commit'), 'helper': ref(Path(__file__)), 'rows': rows, 'summary': dict(Counter(r['status'] for r in rows)), 'benchmark_credit': 0, 'production_authorized': False}
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'audit': ref(path), 'summary': report['summary'], 'errors': [r for r in rows if r['status'] == 'reader_error_requires_diagnosis'][:3]}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('execution', 'output', 'runtime'):
        parser.add_argument('--' + name, required=True)
    main(parser.parse_args())
