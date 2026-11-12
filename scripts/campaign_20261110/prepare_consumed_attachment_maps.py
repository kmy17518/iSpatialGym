"""Author consumed tabletop maps from exact positive attachment-free libraries."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


def support_identity_matches(selected, measured):
    """A profile key alone does not prove another placed support's height."""
    return all(selected.get(a) == measured.get(b) for a, b in (
        ('selected_scale', 'absolute_scale'), ('object', 'object'), ('scene', 'scene')))


def prepare(root, runtime, graph_path, audit_path, output):
    from ispatialgym.grasping import contact_policy as CP, schema as S
    from ispatialgym.sampling import reach_maps as RM
    root, runtime, graph_path, audit_path, output = map(Path, (root, runtime, graph_path, audit_path, output))
    digest_cache = {}
    def sha(path):
        path = str(path)
        if path not in digest_cache: digest_cache[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return digest_cache[path]
    ref = lambda p: {'path': str(p), 'sha256': sha(p)}
    read = lambda p: json.loads(Path(p).read_text())
    graph = read(graph_path); audit = read(audit_path)
    assert audit['schema'] == 'isg-consumed-attachment-native-reader-v1' and not output.exists()
    positives = {}
    for row in audit['rows']:
        if row['status'] != 'verified_positive': continue
        assert row['native_reader_passed'] and row['positive_grasps'] > 0 and sha(row['library']['path']) == row['library']['sha256']
        entry = row['entry']; key = (entry['category'], entry['model'], tuple(S.canonical_scale(entry['scale'])))
        positives[key] = row
    supports = {}
    for row in graph['rows']:
        if row['kind'] != 'native_support_geometry_and_height' or row['status'] != 'verified': continue
        proof = row['artifact']; assert sha(proof['path']) == proof['sha256']
        value = read(proof['path']); assert value['geometry_admitted']
        identity = (row['inputs']['support_profile'], row['inputs']['height_slot'])
        supports[identity] = (proof, value)
    generator = {name: sha(runtime / name) for name in ['ispatialgym/sampling/reach_maps.py', 'ispatialgym/grasping/schema.py',
                 'ispatialgym/grasping/contact_policy.py', 'ispatialgym/grasping/hand_geometry.py', 'ispatialgym/grasping/object_geometry.py']}
    grouped = {}; missing = Counter(); unresolved_supports = {}
    for dep in graph['rows']:
        if dep['kind'] != 'tabletop_reach_map' or not any(c.startswith('configuration_matching-attachment_') for c in dep['required_by_classes']): continue
        if dep['inputs']['arm'] != 'right':
            missing['retired_no_left_tabletop_consumer'] += 1
            continue
        target = dep['inputs']['target']; scale = target.get('applied_scale')
        if not scale: missing['target_scale_missing'] += 1; continue
        source = positives.get((target['category'], target['model'], tuple(S.canonical_scale(scale))))
        if not source: missing['positive_attachment_library_missing'] += 1; continue
        support = dep['inputs']['support']; measured = supports.get((support['support_profile'], support['height_slot']))
        if measured is None: missing['support_height_missing'] += 1; continue
        proof, value = measured
        if not support_identity_matches(support, value):
            missing['support_geometry_identity_unresolved'] += 1
            unresolved_supports[(support['support_profile'], support['height_slot'], support['scene'], support['object'])] = {
                'selected': support, 'measured': {k: value.get(k) for k in ('absolute_scale', 'object', 'scene', 'actual_height_key')}, 'proof': proof}
            continue
        height = value['actual_height_key']; group_key = (source['native_identity_key'], source['library']['sha256'], height)
        unit = grouped.setdefault(group_key, {'source': source, 'height': height, 'arms': set(), 'proofs': {}, 'configuration_ids': set(), 'dependency_ids': set(), 'class_ids': set()})
        unit['arms'].add(dep['inputs']['arm']); unit['proofs'][proof['path']] = proof
        unit['configuration_ids'].update(dep['consumer_scopes']['required']); unit['dependency_ids'].add(dep['id']); unit['class_ids'].update(dep['required_by_classes'])
    # New attachment identity has no reusable maps unless an exact stored entry
    # can prove otherwise. Existing candidate artifacts are retained for a reader.
    wanted_entries = {r['entry']['id'] for r in positives.values()}; candidate_maps = []
    for base in [root / 'training_preparation'] + sorted(root.glob('v2_release*')):
        if not base.is_dir(): continue
        for directory, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if not d.startswith(('floating_', 'camera_')) and d not in ('sampling_cache', 'asset_cache', 'logs', 'jobs', 'native_reader_r1')]
            if Path(directory).name in wanted_entries:
                for name in files:
                    if name.startswith('map__') and name.endswith('.json'): candidate_maps.append(ref(Path(directory) / name))
    output.mkdir(parents=True)
    cache = output / 'exact_cache_inventory.json'
    cache.write_text(json.dumps({'observed_at': time.time(), 'entry_ids': sorted(wanted_entries), 'candidate_maps': candidate_maps,
                                'source_audit': ref(audit_path), 'new_library_identity_only': True}, indent=2) + '\n')
    assert not candidate_maps, 'Existing exact entry maps require native-reader join before new admission'
    runner = '''import hashlib,json,os,subprocess,sys,time
from pathlib import Path
D=Path(__file__).parent;p=json.loads((D/'prepared.json').read_text());sha=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
for r in p['input_refs']:assert sha(r['path'])==r['sha256'],r['path']
v={'started_at':time.time(),'job_id':os.environ['SLURM_JOB_ID'],'physical_dedup_key':p['physical_dedup_key'],'prepared_sha256':sha(D/'prepared.json'),'native_runtime':p['native_runtime'],'benchmark_credit':0,'arms':[]}
(D/'started.json').write_text(json.dumps(v,indent=2)+'\\n')
for arm,original in p['native_argv_by_arm'].items():
 argv=list(original);argv[0]=sys.executable;started=time.time()
 with(D/('native_'+arm+'.log')).open('w')as log:
  try:rc=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=1500).returncode
  except subprocess.TimeoutExpired:rc=124
 done='GB_MAP_DONE'in(D/('native_'+arm+'.log')).read_text(errors='replace')
 v['arms'].append({'arm':arm,'returncode':rc,'native_done':done,'seconds':time.time()-started})
v.update(finished_at=time.time(),map_files=[{'path':str(x),'sha256':sha(x)}for x in(D/'maps').rglob('map__*')if x.is_file()],native_reader_required=True,production_authorized=False)
(D/'terminal.json').write_text(json.dumps(v,indent=2)+'\\n');sys.exit(0 if all(a['returncode']==0 and a['native_done']for a in v['arms'])and v['map_files']else 2)
'''
    compile(runner, 'run.py', 'exec'); tickets = []; units = []
    for _, unit in sorted(grouped.items()):
        row = unit['source']; entry = row['entry']; arms = sorted(unit['arms']); assert arms and set(arms) <= {'left', 'right'}
        physical = {'kind': 'tabletop', 'library_identity': row['native_identity_key'], 'library_sha256': row['library']['sha256'],
                    'height': float(unit['height']), 'arms': arms, 'config': {a: RM.MapConfig(arm=a).key() for a in arms},
                    'robot': RM.robot_digest(), 'policy': {a: CP.DEFAULT.key(a) for a in arms}, 'generator_files': generator}
        key = hashlib.sha256(json.dumps(physical, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        ident = 'attachment_tabletop_' + key[:20]; directory = output / ident; directory.mkdir()
        jobs = directory / 'jobs.json'; jobs.write_text(json.dumps([entry], indent=2) + '\n')
        anchors = directory / 'anchors.json'; height = float(unit['height'])
        anchors.write_text(json.dumps({'schema': 'isg-height-anchors/1', 'anchors': [{'height': height, 'key': unit['height'], 'precision': .0001,
                           'range': [height - .00005, height + .00005], 'status': ['measured'], 'surfaces': sorted(unit['proofs'])}],
                           'max_approx_m': .00005}, indent=2) + '\n')
        selected = directory / 'consumed_inputs.json'; selected.write_text(json.dumps({'entry': entry, 'physical_identity': physical,
                   'configuration_ids': sorted(unit['configuration_ids']), 'dependency_ids': sorted(unit['dependency_ids']),
                   'class_ids': sorted(unit['class_ids']), 'source_graph': ref(graph_path), 'native_library_audit': ref(audit_path),
                   'native_library_evidence': row, 'height_proofs': list(unit['proofs'].values())}, indent=2) + '\n')
        inputs = row['files'] + list(unit['proofs'].values()) + [ref(jobs), ref(anchors), ref(selected), ref(cache), ref(runtime / '.snapshot_commit')]
        inputs += [{'path': str(runtime / n), 'sha256': h} for n, h in generator.items()]
        for r in inputs: assert sha(r['path']) == r['sha256']
        glib = str(Path(row['library']['path']).parents[2])
        argv = {a: ['python', '-u', '-m', 'ispatialgym.sampling.reach_maps', 'compute', '--kind', 'tabletop', '--jobs', str(jobs),
                     '--glib', glib, '--anchors', str(anchors), '--heights', unit['height'], '--arm', a, '--shard', '0/1', '--out', str(directory / 'maps'),
                     '--mode', 'full', '--lattice', '0.08', '--seeds', '8', '--trunk-eligibility', 'per_d_e'] for a in arms}
        pp = directory / 'prepared.json'; pp.write_text(json.dumps({'id': ident, 'physical_dedup_key': key, 'physical_identity': physical,
                'native_runtime': str(runtime), 'native_argv_by_arm': argv, 'input_refs': inputs, 'configuration_ids': sorted(unit['configuration_ids']),
                'entry_id': entry['id'], 'height': unit['height'], 'benchmark_credit': 0, 'production_authorized': False}, indent=2) + '\n')
        (directory / 'run.py').write_text(runner)
        env = {'ISG_GB_ROOT': str(runtime), 'GBH': str(root), 'KIT_TPL_NFS_ROOT': str(root / 'kit_templates'),
               'KIT_TEX_STORE': str(root / 'kit_textures'), 'EXP_PATH': str(root / 'isaacsim_overlay/apps')}
        shell = '#!/bin/bash\nset -euo pipefail\n' + '\n'.join('export ' + k + '=' + shlex.quote(v) for k, v in env.items())
        shell += '\nset +u\nsource "$ISG_GB_ROOT/scripts/grasp_bands/env_cluster.sh"\nset -u\nexport PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR="$JB/numba_cache"\nKIT_RC=1\ntrap \'kit_appdata_teardown\' EXIT\npython -u ' + shlex.quote(str(directory / 'run.py')) + '\nKIT_RC=0\n'
        (directory / 'job.sh').write_text(shell); (directory / 'pinned.sh').write_text('#!/bin/bash\nset -euo pipefail\nexec srun --cpu-bind=cores --exact -n1 -c8 bash ' + shlex.quote(str(directory / 'job.sh')) + '\n')
        subprocess.run(['bash', '-n', str(directory / 'job.sh')], check=True)
        namespace = root / 'infra/coordination/preparation_dispatch/consumed_asset_builds_r1'
        tickets.append({'id': ident, 'physical_dedup_key': key, 'dedup_key': key, 'expected_account': 'minyeong',
            'claim_path': str(namespace / 'claims' / (key + '.json')), 'submission_path': str(namespace / 'receipts' / (key + '.json')),
            'sbatch_argv': ['sbatch', '--parsable', '--nodes=1', '--ntasks=1', '-A', 'vision', '-p', 'svl,viscam', '--exclude=svl9', '--gres=gpu:titanrtx:1', '-c', '8', '--mem=48G', '--time=00:30:00', '--nice=1000', '-J', 'isg261110_attach_maps', '-o', str(directory / 'slurm_%j.log'), str(directory / 'pinned.sh')],
            'prepared': ref(pp), 'runtime_ref': ref(runtime / '.snapshot_commit'), 'input_refs': inputs,
            'scripts': [ref(directory / n) for n in ('run.py', 'job.sh', 'pinned.sh')], 'output': str(directory / 'maps'),
            'dispatch_workdir': str(runtime), 'hard_time_seconds': 1800, 'native_time_seconds': 1500,
            'computation_kind': 'reach_map', 'consumer_kind': 'tabletop', 'workstream': 'training preparation', 'owner': '/root/templates_assets',
            'stage': 'attachment male tabletop maps', 'class_ids': sorted(unit['class_ids']), 'configuration_ids': sorted(unit['configuration_ids']),
            'selected_dependency_ids': sorted(unit['dependency_ids']), 'followup': 'Native reader checks exact library/rest/height/arm/config/robot/policy and grid coverage; preserve valid zeros; no scene query.',
            'benchmark_credit': 0, 'production_authorized': False})
        units.append({'id': ident, 'physical_dedup_key': key, 'entry_id': entry['id'], 'height': unit['height'], 'arms': arms,
                      'logical_requirements': len(unit['dependency_ids']), 'configuration_ids': sorted(unit['configuration_ids'])})
    p = output / 'tickets.json'; p.write_text(json.dumps({'schema': 'isg-preparation-dispatch-tickets-v1', 'campaign_id': root.name,
        'account': 'minyeong', 'owner': '/root/templates_assets', 'dispatch_owner': '/root', 'observed_at': time.time(), 'jobs': tickets,
        'source_graph': ref(graph_path), 'source_library_audit': ref(audit_path), 'missing_logical_requirements': dict(missing),
        'unresolved_support_identities': list(unresolved_supports.values()),
        'units': units, 'benchmark_credit': 0, 'production_authorized': False}, indent=2) + '\n')
    print(json.dumps({'manifest': ref(p), 'physical_units': len(units), 'arm_queries': sum(len(u['arms']) for u in units),
                      'logical_requirements': sum(u['logical_requirements'] for u in units), 'missing': dict(missing)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'runtime', 'graph', 'audit', 'output'): parser.add_argument('--' + key, required=True)
    args = parser.parse_args(); prepare(args.root, args.runtime, args.graph, args.audit, args.output)
