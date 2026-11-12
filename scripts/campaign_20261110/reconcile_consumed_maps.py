"""Finite native map-verification callback for the sole root reconciler.

No dispatch, provider action, source-output modification, or scene sampling.
Terminal, artifact, verification and submission timestamps stay distinct.
"""
import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import signal
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
    if ticket['computation_kind'] == 'contact_map':
        return native_read_contact(ticket, terminal_path, maps_root, backend, cache)
    if ticket['consumer_kind'] == 'generic':
        return native_read_generic(ticket, terminal_path, maps_root, backend, cache)
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
    lib = load_map_library(S, entry, args['--glib'])
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


def load_map_library(schema, entry, root):
    return schema.load_model(entry['category'], entry['model'], entry['scale'], root,
                             constraint=entry['constraint'], rest_hint_wxyz=entry.get('rest_hint_wxyz'))


def contact_grid_proof(native, identity, library, cm, cp, rm, np):
    """Check the exact required push-map semantics before consumer lookup."""
    meta = native.meta
    require(identity['kind'] == 'push_blade' and identity['arm'] == 'right', 'Unsupported required contact identity')
    require(meta['schema'] == cm.SCHEMA and meta['key'] == cm.map_key(meta), 'Contact native map key differs')
    require(meta['kind'] == identity['kind'] and meta['arm'] == identity['arm'] and meta['plane'] == cm.KINDS[identity['kind']]['plane'], 'Contact kind/arm/plane differs')
    require(cm.object_key(meta['object']) == identity['object_key'] == cm.object_key(library['object']), 'Contact object identity differs')
    require(meta['library']['digest'] == library['digest'], 'Contact library digest differs')
    require(meta['height']['value'] == identity['height_above_floor'] and meta['pose_set'] == identity['pose_set'], 'Contact height/pose set differs')
    normalize = lambda value: json.loads(json.dumps(value))
    require(meta['config'] == normalize(identity['config']) and meta['edge_distances'] == normalize(identity['config']['edge_distances']), 'Contact configuration differs')
    require(meta['policy_key'] == identity['policy_key'] == cp.DEFAULT.key('right') and meta['robot'] == identity['robot'] == rm.robot_digest(), 'Contact policy/robot differs')
    require(np.isfinite(native.cells).all() and np.isfinite(native.yaws).all(), 'Nonfinite contact grid')
    grids = [v for k, v in native.arrays.items() if k.startswith('s')]
    allowed = set(cm.STATUS.values()) | {cm.STATUS_IK_MARGIN}
    require(grids and all(set(np.unique(v)) <= allowed for v in grids), 'Invalid contact status grid')
    require(all(v.shape == (len(native.cells), len(native.yaws), len(native.classes)) for v in grids), 'Contact grid dimensions differ')
    return sum(int((v == cm.STATUS['feasible']).sum()) for v in grids)


def native_read_contact(ticket, terminal_path, maps_root, backend, cache):
    import numpy as np
    from ispatialgym.grasping import contact_library as CL, contact_policy as CP
    from ispatialgym.sampling import contact_maps as CM, reach_maps as RM
    from ispatialgym.sampling.v2 import contact_poses as CPZ
    require(backend == 'cluster', 'Required contact transport currently supports cluster artifacts only')
    prepared = checked(ticket['prepared'], cache)
    for ref in ticket['input_refs'] + [ticket['runtime_ref']]: checked_hash(ref, cache)
    for module in (CL, CP, CM, RM, CPZ):
        expected = next(r for r in ticket['input_refs'] if Path(r['path']).name == Path(module.__file__).name)
        require(sha(module.__file__) == expected['sha256'], 'Native contact reader runtime differs')
    terminal = json.loads(terminal_path.read_text()); identity = prepared['physical_identity']
    require(terminal['physical_dedup_key'] == ticket['physical_dedup_key'] and terminal['prepared_sha256'] == ticket['prepared']['sha256'], 'Contact terminal physical identity differs')
    require(terminal['returncode'] == 0 and terminal['native_done'], 'Native contact builder failed')
    source = prepared['source_library']; checked_hash(source, cache)
    library = CL.load_library(source['path'])
    require(library['digest'] == CL._library_digest(library), 'Contact library payload digest differs')
    entries = CM.select_entries(library, identity['kind'], ('valid',), arm=identity['arm'])
    require(entries and CM.pose_set(identity['kind'], entries, ('valid',)) == identity['pose_set'], 'Contact positive entry pose set differs')
    paths = sorted(maps_root.rglob('cmap__*.json')); require(len(paths) == 1, 'One required physical contact map expected')
    path = paths[0]; native = CM.ContactMap(str(path))
    positive = contact_grid_proof(native, identity, library, CM, CP, RM, np)
    found = CM.ContactMap.find(str(maps_root), identity['kind'], identity['object_key']['entry'], identity['height_above_floor'],
        rest_id=library['object']['rest_id'], arm=identity['arm'], policy_key=identity['policy_key'], robot=identity['robot'],
        config_require=identity['config'], pose_set_digest=identity['pose_set']['digest'])
    require(found is not None and Path(found.path) == path, 'Native contact consumer lookup failed')
    declared = {r['path']: r['sha256'] for r in terminal['map_files']}
    files = [{'path': str(p), 'sha256': sha(p)} for p in (path, path.with_suffix('.npz'))]
    require(all(declared.get(r['path']) == r['sha256'] for r in files), 'Unbound contact map payload')
    return {'status': 'verified_positive' if positive else 'valid_zero', 'native_reader_passed': True,
            'positive_cells': positive, 'verified_files': files,
            'maps': [{'arm': identity['arm'], 'pose_set': identity['pose_set'], 'files': files}],
            'builder_elapsed_seconds': terminal['finished_at'] - terminal['started_at'],
            'downstream_computation_required': False, 'configuration_readiness_claimed': False, 'benchmark_credit': 0,
            'next_action': 'Reuse exact required contact map in configuration data closure' if positive else 'Diagnose valid zero; do not repeat unchanged'}


def native_read_generic(ticket, terminal_path, maps_root, backend, cache):
    import numpy as np
    from ispatialgym.grasping import contact_policy as CP
    from ispatialgym.sampling import reach_maps as RM
    require(backend == 'cluster', 'Generic transport currently supports cluster artifacts only')
    prepared = checked(ticket['prepared'], cache)
    for ref in ticket['input_refs'] + [ticket['runtime_ref']]: checked_hash(ref, cache)
    for module in (CP, RM):
        expected = next(r for r in ticket['input_refs'] if Path(r['path']).name == Path(module.__file__).name)
        require(sha(module.__file__) == expected['sha256'], 'Native generic reader runtime differs')
    terminal = json.loads(terminal_path.read_text()); identity = prepared['physical_identity']
    require(terminal['physical_dedup_key'] == ticket['physical_dedup_key'] and terminal['prepared_sha256'] == ticket['prepared']['sha256'], 'Generic terminal physical identity differs')
    require(terminal['returncode'] == 0 and terminal['native_done'], 'Native generic builder failed')
    declared = {r['path']: r['sha256'] for r in terminal['map_files']}
    paths = sorted(maps_root.rglob('map__*.json')); require(len(paths) == 1, 'One generic physical map required')
    path = paths[0]; native = RM.ReachMap(str(path)); meta = native.meta
    require(meta['kind'] == 'generic' and meta['key'] == RM.generic_map_key(meta), 'Generic key differs')
    require(meta['arm'] == identity['arm'] and meta['pose_set'] == identity['pose_set'] and meta['config'] == identity['config'], 'Generic pose or configuration differs')
    require(meta['height']['value'] == identity['height'] and meta['robot'] == identity['robot'] == RM.robot_digest(), 'Generic height/robot differs')
    require(meta['policy_key'] == identity['policy'] == CP.DEFAULT.key(identity['arm']), 'Generic policy differs')
    require(np.isfinite(native.cells).all() and np.isfinite(native.yaws).all(), 'Nonfinite generic grid')
    status = [v for k, v in native.arrays.items() if k.startswith('s')]
    require(status and all(set(np.unique(v)) <= set(RM.STATUS.values()) for v in status), 'Invalid generic status grid')
    hit = RM.find_generic(str(path.parent), identity['height'], arm=identity['arm'], pose_set=identity['pose_set'],
                          policy_key=identity['policy'], robot=identity['robot'], config_require=identity['config'])
    require(hit is not None and Path(hit.path) == path, 'Native generic consumer lookup failed')
    files = [{'path': str(p), 'sha256': sha(p)} for p in (path, path.with_suffix('.npz'))]
    require(all(declared.get(r['path']) == r['sha256'] for r in files), 'Unbound generic map payload')
    positive = sum(int((v == RM.STATUS['feasible']).sum()) for v in status)
    return {'status': 'verified_positive' if positive else 'valid_zero', 'native_reader_passed': True,
            'positive_cells': positive, 'maps': [{'arm': identity['arm'], 'pose_set': identity['pose_set'], 'files': files}],
            'verified_files': files, 'builder_elapsed_seconds': terminal['finished_at'] - terminal['started_at'],
            'downstream_computation_required': False, 'configuration_readiness_claimed': False, 'benchmark_credit': 0,
            'next_action': 'Reuse exact generic map in configuration data closure' if positive else 'Diagnose valid zero; do not repeat unchanged'}


def checked_hash(ref, cache):
    if ref['path'] not in cache: cache[ref['path']] = sha(ref['path'])
    require(cache[ref['path']] == ref['sha256'], 'Input changed: ' + ref['path'])


def account_root(account):
    return Path('/vision/u/shiyuc/minyeong/campaign_20261110' if account == 'shiyuc' else
                '/vision/u/minyeong/projects/isg_home/campaign_20261110')


def peer_verification(ticket, receipt, account, cache):
    """Project an owning-account native proof; private map payloads stay there."""
    executing = receipt.get('expected_account', ticket.get('expected_account', 'minyeong'))
    if receipt.get('backend') == 'vast': executing = 'minyeong'
    if executing == account: return None
    require(executing in {'minyeong', 'shiyuc'}, 'Unknown executing account')
    base = account_root(executing)
    peer = base / 'infra/coordinator_dispatch/preparation_verification' / (ticket['physical_dedup_key'] + '.json')
    row = {'status': 'delegated_verification_pending', 'verification_account': executing,
           'peer_sidecar_path': str(peer), 'blocker': 'Owning-account native verifier has not published a terminal proof',
           'verified_files': []}
    # Secondary workers never import primary proofs or write a competing sidecar.
    if account != 'minyeong' or not peer.exists(): return row
    side = json.loads(peer.read_text())
    require(side['physical_dedup_key'] == ticket['physical_dedup_key'], 'Peer verification key differs')
    if side['status'] not in {'verified_positive', 'verified_zero', 'failed'}: return row
    if side['status'] == 'failed':
        return {**row, 'status': 'peer_native_verification_failed', 'blocker': side.get('blocker'), 'peer_sidecar': {'path': str(peer), 'sha256': sha(peer)}}
    proof_ref = next((r for r in side['evidence'] if '/verifications/' in r['path']), None)
    require(proof_ref is not None and Path(proof_ref['path']).is_relative_to(base), 'Peer native proof missing')
    proof = checked(proof_ref, cache)
    primary_prepared = checked(ticket['prepared'], cache); peer_prepared = checked(receipt['prepared'], cache)
    require(primary_prepared['physical_identity'] == peer_prepared['physical_identity'], 'Peer physical input identity differs')
    require(proof['physical_dedup_key'] == ticket['physical_dedup_key'] and proof.get('native_reader_passed') is True and
            str(proof['job_id']) == str(receipt['job_id']), 'Peer proof execution identity differs')
    require(proof['status'] in {'verified_positive', 'valid_zero'}, 'Peer proof disposition differs')
    terminal = checked(proof['terminal'], cache)
    require(terminal['physical_dedup_key'] == ticket['physical_dedup_key'] and
            terminal['prepared_sha256'] == receipt['prepared']['sha256'], 'Peer terminal preparation differs')
    require(not (Path(ticket['prepared']['path']).parent / 'terminal.json').exists(), 'Both account outputs exist for one physical claim')
    return {**row, **{k: proof[k] for k in ('status', 'native_reader_passed', 'positive_cells', 'verification_at', 'verification_started_at', 'terminal_at', 'terminal')},
            'verification': proof_ref, 'peer_verified_files': proof['verified_files'], 'peer_sidecar': {'path': str(peer), 'sha256': sha(peer)},
            'blocker': None, 'configuration_readiness_claimed': False, 'benchmark_credit': 0}


def sidecar(root, row, *, running=False):
    status = ('running' if running else {'valid_zero': 'verified_zero', 'invalid_or_failed': 'failed',
              'observation_unavailable': 'failed', 'peer_native_verification_failed': 'failed', 'computed_unverified': 'pending'}.get(row['status'], row['status']))
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


def reconcile(root, manifest_ref, route_ref, output, *, max_seconds=15, max_items=128, reader=native_read, timestamp=None):
    root, output = Path(root), Path(output)
    account = 'shiyuc' if str(root).startswith('/vision/u/shiyuc/') else 'minyeong'
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
            require((ticket['computation_kind'], ticket['consumer_kind']) in {('reach_map', 'tabletop'), ('reach_map', 'generic'), ('contact_map', 'push_blade')}, 'Unsupported callback kind')
            receipt = Path(ticket['submission_path'])
            if receipt.exists():
                value = json.loads(receipt.read_text()); row['submission'] = {'path': str(receipt), 'sha256': sha(receipt)}
                row['submitted_at'] = epoch(value.get('submitted_at') or value.get('observed_at'))
                row['job_id'] = value.get('job_id')
                delegated = peer_verification(ticket, value, account, cache)
                if delegated is not None:
                    row.update(delegated)
                    state['rows'][key] = row; rows.append(row)
                    if account == 'minyeong': sidecar(root, row)
                    continue
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


def run_cycles(callback, *, interval, max_cycles, stopped, clock=time):
    """Start-based cadence: a long scan never adds another full sleep interval."""
    for index in range(max_cycles):
        if stopped[0]: break
        started = clock.monotonic(); callback()
        if index + 1 >= max_cycles: break
        while not stopped[0] and clock.monotonic() < started + interval:
            clock.sleep(max(0, min(1, started + interval - clock.monotonic())))


def verifier_registry(root, pointer):
    """Root may atomically point to a new immutable set of exact data inputs."""
    root = Path(root).resolve(); pointer = Path(pointer).resolve()
    require(pointer.is_relative_to(root), 'Verifier registry pointer outside campaign')
    ref = json.loads(pointer.read_text())
    require(Path(ref['path']).resolve().is_relative_to(root), 'Verifier registry outside campaign')
    value = checked(ref, {})
    require(value['schema'] == 'isg-consumed-map-verifier-inputs-v1' and value['owner'] == '/root', 'Unsupported verifier registry')
    require(len(value['manifests']) <= 64, 'Unbounded verifier registry')
    for item in value['manifests']:
        require(Path(item['manifest']['path']).resolve().is_relative_to(root), 'Additional manifest outside campaign')
        require(Path(item['output']).resolve().is_relative_to(root), 'Additional output outside campaign')
    return value['manifests']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'manifest', 'manifest-sha256', 'output'): parser.add_argument('--' + key, required=True)
    parser.add_argument('--route'); parser.add_argument('--route-sha256')
    parser.add_argument('--library-audit'); parser.add_argument('--library-audit-sha256')
    parser.add_argument('--additional-manifests'); parser.add_argument('--additional-manifests-sha256')
    parser.add_argument('--additional-manifests-pointer')
    parser.add_argument('--max-seconds', type=int, default=15); parser.add_argument('--max-items', type=int, default=128)
    parser.add_argument('--watch', action='store_true'); parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--max-cycles', type=int, default=720)
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    require(30 <= args.interval <= 60 and 1 <= args.max_cycles <= 2880, 'Unbounded watch cadence/lifetime')
    stopped = [False]
    additional = []
    additional_cursor = [0]
    require(not (args.additional_manifests and args.additional_manifests_pointer), 'Choose immutable list or root registry pointer')
    if args.additional_manifests:
        extra = checked({'path': args.additional_manifests, 'sha256': args.additional_manifests_sha256}, {})
        require(extra['schema'] == 'isg-consumed-map-verifier-inputs-v1', 'Unsupported extra verifier inputs')
        additional = extra['manifests']
        require(len(additional) <= 64, 'Unbounded extra verifier manifests')
    def stop(signum, frame): stopped[0] = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)

    def run():
        began = time.monotonic()
        current_additional = verifier_registry(args.root, args.additional_manifests_pointer) if args.additional_manifests_pointer else additional
        with atomic_directory_lock(output / 'callback.lockdir'):
            report = reconcile(args.root, {'path': args.manifest, 'sha256': args.manifest_sha256},
                               {'path': args.route, 'sha256': args.route_sha256} if args.route else None, args.output,
                               max_seconds=args.max_seconds, max_items=args.max_items)
            if args.library_audit:
                project_library_advancement(args.root, {'path': args.manifest, 'sha256': args.manifest_sha256},
                                            {'path': args.library_audit, 'sha256': args.library_audit_sha256})
            print(json.dumps({k: report[k] for k in ('summary', 'verified_this_cycle', 'cycle_seconds')}), flush=True)
            start = additional_cursor[0] % max(1, len(current_additional))
            for offset in range(len(current_additional)):
                if time.monotonic() - began >= args.max_seconds: break
                item = current_additional[(start + offset) % len(current_additional)]
                target = Path(item['output'])
                require(target.resolve().is_relative_to(Path(args.root).resolve()) and target != output, 'Invalid additional verifier output')
                target.mkdir(parents=True, exist_ok=True)
                with atomic_directory_lock(target / 'callback.lockdir'):
                    extra = reconcile(args.root, item['manifest'], item.get('route'), target,
                                      max_seconds=max(1, args.max_seconds - int(time.monotonic() - began)), max_items=args.max_items)
                    print(json.dumps({'additional_manifest': item['manifest'], **{k: extra[k] for k in ('summary', 'verified_this_cycle', 'cycle_seconds')}}), flush=True)
                    additional_cursor[0] = start + offset + 1
            return report

    try:
        with atomic_directory_lock(output / ('watch_owner.lockdir' if args.watch else 'finite_owner.lockdir')):
            startup = time.time()
            write(output / 'worker_startup.json', {'started_at': startup, 'owner': '/root/templates_assets',
                  'execution_owner': '/root', 'job_id': os.environ.get('SLURM_JOB_ID'),
                  'phase': 'importing_native_reader', 'dispatch_authority': False})
            from ispatialgym.grasping import schema as S, contact_policy as CP
            from ispatialgym.sampling import reach_maps as RM
            RM.robot_digest(); CP.DEFAULT.key('right')
            write(output / 'worker_startup.json', {'started_at': startup, 'finished_at': time.time(),
                  'cold_start_seconds': time.time() - startup, 'owner': '/root/templates_assets',
                  'execution_owner': '/root', 'job_id': os.environ.get('SLURM_JOB_ID'), 'phase': 'native_reader_ready'})
            def attempt():
                try: run()
                except FileExistsError:
                    print(json.dumps({'callback': 'scan_lock_owned_elsewhere', 'output': args.output}), flush=True)
            run_cycles(attempt, interval=args.interval, max_cycles=args.max_cycles if args.watch else 1, stopped=stopped)
    except FileExistsError:
        print(json.dumps({'callback': 'existing_lock_requires_owner_reconciliation', 'output': args.output})); return


if __name__ == '__main__': main()
