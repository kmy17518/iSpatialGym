"""Finite native map-verification callback for the sole root reconciler.

No dispatch, provider action, source-output modification, or scene sampling.
Terminal, artifact, verification and submission timestamps stay distinct.
"""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import time

from scripts.campaign_20261110.vast_prebuild import require, sha, write
from scripts.campaign_20261110.storage import atomic_directory_lock


def epoch(value):
    if isinstance(value, (int, float)): return float(value)
    if isinstance(value, str):
        try: return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
        except ValueError: return None
    return None


def checked(ref, cache):
    path = ref['path']
    if path not in cache: cache[path] = sha(path)
    require(cache[path] == ref['sha256'], 'Input changed: ' + path)
    return json.loads(Path(path).read_text())


def locate_terminal(ticket, route):
    directory = Path(ticket['prepared']['path']).parent
    candidates = [(directory / 'terminal.json', directory / 'maps', 'cluster')]
    if route:
        job = Path(route['job_dir'])
        candidates.append((job / 'result/map_prebuild_result.json', job / 'result/maps', 'vast'))
    present = [x for x in candidates if x[0].exists()]
    require(len(present) <= 1, 'Two backend terminal outputs for one physical claim')
    return present[0] if present else None


def native_read(ticket, terminal_path, maps_root, backend, route, cache):
    import numpy as np
    from ispatialgym.grasping import schema as S, contact_policy as CP
    from ispatialgym.sampling import reach_maps as RM
    prepared = checked(ticket['prepared'], cache)
    for ref in ticket['input_refs'] + [ticket['runtime_ref']]: checked_hash(ref, cache)
    for module in (S, CP, RM):
        expected = next(r for r in ticket['input_refs'] if Path(r['path']).name == Path(module.__file__).name)
        require(sha(module.__file__) == expected['sha256'], 'Native reader runtime differs')
    terminal = json.loads(terminal_path.read_text())
    if backend == 'cluster':
        require(terminal['physical_dedup_key'] == ticket['physical_dedup_key'] and terminal['prepared_sha256'] == ticket['prepared']['sha256'], 'Terminal physical identity differs')
        require(all(r['returncode'] == 0 and r['native_done'] for r in terminal['arms']), 'Native map builder failed')
        declared = {r['path']: r['sha256'] for r in terminal['map_files']}
    else:
        value = checked(route['native_inputs'], cache)
        require(value['physical_dedup_key'] == ticket['physical_dedup_key'] and value['configuration_ids'] == ticket['configuration_ids'], 'Vast physical identity differs')
        require(all(r['returncode'] == 0 and r['native_map_done'] for r in terminal['arms']), 'Native Vast map builder failed')
        declared = terminal['map_files']
    commands = prepared['native_argv_by_arm']; arms = sorted(commands)
    argv = next(iter(commands.values())); args = dict(zip(argv[7::2], argv[8::2]))
    jobs = json.loads(Path(args['--jobs']).read_text()); require(len(jobs) == 1, 'One physical map library required')
    entry = jobs[0]
    lib = S.load_model(entry['category'], entry['model'], entry['scale'], args['--glib'], constraint=entry['constraint'])
    require(lib is not None and lib.grasps(), 'Map library must have positive native grasps')
    identity = prepared['physical_identity']
    require(lib.identity_key() == identity['library_identity'] and sha(lib.path) == identity['library_sha256'], 'Map library identity differs')
    expected = {(g['rest_id'], arm) for g in lib.grasps() for arm in arms}; actual = set(); rows = []; files = []
    for path in sorted(maps_root.rglob('map__*.json')):
        require(str(path) in declared and sha(path) == declared[str(path)], 'Unbound terminal map payload')
        native = RM.ReachMap(str(path)); meta = native.meta; arm = meta['arm']; rest = meta['rest_id']
        require(meta['kind'] == 'tabletop' and meta['key'] == RM.map_key(meta) and meta['object_key'] == lib.identity_key(), 'Native map identity differs')
        require((rest, arm) in expected and (rest, arm) not in actual, 'Map coverage differs'); actual.add((rest, arm))
        require(meta['height']['value'] == identity['height'] and meta['robot'] == RM.robot_digest() and meta['policy_key'] == CP.DEFAULT.key(arm), 'Map height/robot/policy differs')
        require(meta['config'] == json.loads(json.dumps(RM.MapConfig(arm=arm).key())), 'Map configuration differs')
        require(np.isfinite(native.cells).all() and np.isfinite(native.yaws).all(), 'Nonfinite map grid')
        status = [v for k, v in native.arrays.items() if k.startswith('s')]
        require(status and all(set(np.unique(v)) <= set(RM.STATUS.values()) for v in status), 'Invalid map status grid')
        positive = sum(int((v == RM.STATUS['feasible']).sum()) for v in status)
        hit = RM.ReachMap.find(str(path.parent), lib, identity['height'], rest_id=rest, arm=arm,
                              policy_key=CP.DEFAULT.key(arm), robot=RM.robot_digest())
        require(hit is not None and Path(hit.path) == path, 'Native map consumer lookup failed')
        closure = [{'path': str(p), 'sha256': sha(p)} for p in (path, path.with_suffix('.npz'))]
        files += closure; rows.append({'arm': arm, 'rest_id': rest, 'positive_cells': positive, 'files': closure})
    require(actual == expected and rows, 'Incomplete native map coverage')
    positive = sum(r['positive_cells'] for r in rows)
    return {'status': 'verified_positive' if positive else 'valid_zero', 'native_reader_passed': True,
            'positive_cells': positive, 'maps': rows, 'verified_files': files,
            'builder_elapsed_seconds': terminal.get('finished_at', 0) - terminal.get('started_at', 0),
            'downstream_computation_required': False,
            'next_action': 'Reuse exact map artifact in configuration data closure' if positive else 'Diagnose valid zero; do not repeat unchanged',
            'configuration_readiness_claimed': False, 'benchmark_credit': 0}


def checked_hash(ref, cache):
    if ref['path'] not in cache: cache[ref['path']] = sha(ref['path'])
    require(cache[ref['path']] == ref['sha256'], 'Input changed: ' + ref['path'])


def sidecar(root, row, *, running=False):
    status = ('running' if running else {'valid_zero': 'verified_zero', 'invalid_or_failed': 'failed',
              'observation_unavailable': 'failed', 'computed_unverified': 'pending'}.get(row['status'], row['status']))
    if status not in {'running', 'pending', 'failed', 'verified_positive', 'verified_zero'}: return
    value = {'physical_dedup_key': row['physical_dedup_key'], 'owner': '/root/templates_assets',
             'terminal_at': row.get('terminal_at'), 'terminal_time_source': 'native_builder_terminal_artifact',
             'verification_started_at': row.get('verification_started_at'), 'verified_at': row.get('verification_at'),
             'status': status, 'evidence': [row[k] for k in ('terminal', 'verification') if row.get(k)],
             'downstream_required': False, 'downstream_ids': [], 'downstream_submitted_at': None,
             'blocker': row.get('blocker'), 'next_check_at': time.time() + 60 if status in {'running', 'pending', 'failed'} else None,
             'configuration_readiness_claimed': False, 'verification_callback': 'scripts.campaign_20261110.reconcile_consumed_maps',
             'benchmark_credit': 0}
    write(Path(root) / 'infra/coordinator_dispatch/preparation_verification' / (row['physical_dedup_key'] + '.json'), value)


def project_library_advancement(root, manifest_ref, audit_ref):
    """Project an existing native audit, without relabeling observation as work."""
    cache = {}; manifest = checked(manifest_ref, cache); audit = checked(audit_ref, cache); by_library = {}
    require(audit['schema'] == 'isg-consumed-attachment-native-reader-v1', 'Unsupported native library audit')
    for ticket in manifest['jobs']:
        prepared = checked(ticket['prepared'], cache)
        by_library.setdefault(prepared['physical_identity']['library_sha256'], []).append(ticket)
    for row in audit['rows']:
        require(row['native_reader_passed'] and row['status'] in {'verified_positive', 'valid_zero'}, 'Library has no verified disposition')
        terminal = checked(row['terminal'], cache); checked_hash(row['library'], cache)
        downstream = by_library.get(row['library']['sha256'], []) if row['status'] == 'verified_positive' else []
        submitted = {}; pending = []
        for ticket in downstream:
            receipt = Path(ticket['submission_path'])
            value = json.loads(receipt.read_text()) if receipt.exists() else {}
            at = epoch(value.get('submitted_at') or value.get('observed_at'))
            if value.get('job_id') and at is not None: submitted[ticket['physical_dedup_key']] = at
            else: pending.append(ticket['physical_dedup_key'])
        missing = row['status'] == 'verified_positive' and not downstream
        value = {'physical_dedup_key': row['physical_dedup_key'], 'owner': '/root/templates_assets',
                 'terminal_at': terminal.get('finished_at'), 'terminal_time_source': 'native_builder_terminal_artifact',
                 'verification_started_at': None, 'verified_at': audit['observed_at'],
                 'status': 'verified_positive' if row['status'] == 'verified_positive' else 'verified_zero',
                 'evidence': [audit_ref, row['terminal'], row['library']], 'downstream_required': row['status'] == 'verified_positive',
                 'downstream_ids': [t['physical_dedup_key'] for t in downstream], 'downstream_submissions': submitted,
                 'downstream_submitted_at': max(submitted.values()) if submitted and not pending else None,
                 'blocker': 'Exact downstream map manifest missing' if missing else 'Ready downstream map tickets await root submission' if pending else None,
                 'next_check_at': time.time() + 60 if missing or pending else None, 'benchmark_credit': 0}
        write(Path(root) / 'infra/coordinator_dispatch/preparation_verification' / (row['physical_dedup_key'] + '.json'), value)


def reconcile(root, manifest_ref, route_ref, output, *, max_seconds=45, max_items=128, reader=native_read, timestamp=None):
    root, output = Path(root), Path(output)
    require(output.resolve().is_relative_to(root.resolve()), 'Callback output outside campaign')
    require(1 <= max_seconds <= 50 and 1 <= max_items <= 512, 'Unbounded callback')
    timestamp = time.time() if timestamp is None else timestamp; cache = {}; manifest = checked(manifest_ref, cache)
    require(manifest['schema'] == 'isg-preparation-dispatch-tickets-v1', 'Unsupported map tickets')
    route = checked(route_ref, cache) if route_ref else {'jobs': []}
    if route_ref: require(route['source_manifest'] == manifest_ref, 'Vast source manifest differs')
    routes = {r['physical_dedup_key']: r for r in route['jobs']}
    output.mkdir(parents=True, exist_ok=True); state_path = output / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'rows': {}, 'cursor': 0}
    rows = []; verified = 0; began = time.monotonic(); jobs = manifest['jobs']; count = len(jobs); start = state['cursor'] % max(1, count)
    for offset in range(count):
        ticket = jobs[(start + offset) % count]; key = ticket['physical_dedup_key']; old = state['rows'].get(key, {})
        row = {**old, 'physical_dedup_key': key, 'configuration_ids': ticket['configuration_ids'],
               'owner': '/root/templates_assets', 'manifest_ready_at': epoch(manifest.get('observed_at')),
               'observed_at': timestamp, 'dispatch_owner': '/root'}
        try:
            require(ticket['computation_kind'] == 'reach_map' and ticket['consumer_kind'] == 'tabletop', 'Unsupported callback kind')
            receipt = Path(ticket['submission_path'])
            if receipt.exists():
                value = json.loads(receipt.read_text()); row['submission'] = {'path': str(receipt), 'sha256': sha(receipt)}
                row['submitted_at'] = epoch(value.get('submitted_at') or value.get('observed_at'))
                row['job_id'] = value.get('job_id')
            found = locate_terminal(ticket, routes.get(key))
            if found is None:
                row['status'] = 'awaiting_terminal' if receipt.exists() else 'claimed_unresolved' if Path(ticket['claim_path']).exists() else 'ready_unsubmitted'
                row['blocker'] = 'Await terminal artifact; scheduler execution state is root-owned' if receipt.exists() else 'Exclusive claim has no terminal/receipt' if Path(ticket['claim_path']).exists() else None
            else:
                terminal_path, maps_root, backend = found; terminal = json.loads(terminal_path.read_text()); terminal_sha = sha(terminal_path)
                row.update(terminal={'path': str(terminal_path), 'sha256': terminal_sha}, backend=backend,
                           terminal_at=epoch(terminal.get('finished_at')) or terminal_path.stat().st_mtime,
                           artifact_ready_at=epoch(terminal.get('finished_at')) or terminal_path.stat().st_mtime)
                proof = output / 'verifications' / key / (terminal_sha + '.json')
                if proof.exists() and old.get('verification_at') is None:
                    retained = json.loads(proof.read_text())
                    require(retained['physical_dedup_key'] == key and retained['terminal']['sha256'] == terminal_sha, 'Retained verification identity differs')
                    for ref in retained.get('verified_files', []): checked_hash(ref, cache)
                    row.update(retained, observed_at=timestamp, verification={'path': str(proof), 'sha256': sha(proof)})
                    old = dict(row)
                same = old.get('terminal', {}).get('sha256') == terminal_sha and (old.get('verification_at') is not None or old.get('status') == 'invalid_or_failed')
                if same:
                    for ref in old.get('verified_files', []): checked_hash(ref, cache)
                elif verified >= max_items or time.monotonic() - began >= max_seconds:
                    row.update(status='computed_unverified', blocker='Bounded native reader callback queue', verification_owner='/root/templates_assets')
                else:
                    row['verification_started_at'] = time.time()
                    sidecar(root, row, running=True)
                    result = reader(ticket, terminal_path, maps_root, backend, routes.get(key), cache); verified += 1
                    row.update(result, verification_at=time.time(), blocker=None)
                    require(not proof.exists(), 'Verification evidence already exists; reconcile retained result')
                    write(proof, row); row['verification'] = {'path': str(proof), 'sha256': sha(proof)}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row.update(status='observation_unavailable' if isinstance(exc, OSError) else 'invalid_or_failed',
                       blocker=type(exc).__name__ + ': ' + str(exc), disposition_at=time.time(), diagnosis_owner='/root/templates_assets')
        state['rows'][key] = row; rows.append(row)
        sidecar(root, row)
    state['cursor'] = start + max(1, verified)
    status = {'schema': 'isg-consumed-map-callback-v1', 'observed_at': timestamp, 'owner': '/root/templates_assets',
              'manifest': manifest_ref, 'route': route_ref, 'rows': rows, 'summary': dict(Counter(r['status'] for r in rows)),
              'verified_this_cycle': verified, 'cycle_seconds': time.monotonic() - began,
              'ready_count': sum(r['status'] == 'ready_unsubmitted' for r in rows),
              'completed_awaiting_verification': sum(r['status'] == 'computed_unverified' for r in rows),
              'completed_awaiting_downstream': 0, 'completed_awaiting_downstream_scope': 'These maps are leaf computations; configuration closure remains separately enumerated.',
              'full_campaign_ready_queue_enumerated': False, 'benchmark_credit': 0, 'production_authorized': False}
    write(state_path, state); write(output / 'status.json', status); return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'manifest', 'manifest-sha256', 'output'): parser.add_argument('--' + key, required=True)
    parser.add_argument('--route'); parser.add_argument('--route-sha256')
    parser.add_argument('--library-audit'); parser.add_argument('--library-audit-sha256'); args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    try:
        with atomic_directory_lock(output / 'callback.lockdir'):
            report = reconcile(args.root, {'path': args.manifest, 'sha256': args.manifest_sha256},
                               {'path': args.route, 'sha256': args.route_sha256} if args.route else None, args.output)
            if args.library_audit:
                project_library_advancement(args.root, {'path': args.manifest, 'sha256': args.manifest_sha256},
                                            {'path': args.library_audit, 'sha256': args.library_audit_sha256})
    except FileExistsError:
        print(json.dumps({'callback': 'existing_lock_requires_owner_reconciliation', 'output': args.output})); return
    print(json.dumps({k: report[k] for k in ('summary', 'verified_this_cycle', 'cycle_seconds')}))


if __name__ == '__main__': main()
