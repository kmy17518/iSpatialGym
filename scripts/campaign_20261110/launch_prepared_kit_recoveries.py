"""Finite root-only launcher for two exact changed startup recoveries.

Default is read-only preflight. Existing physical claims are never rewritten;
a durable attempt intent prevents ambiguous submissions from being retried.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time


def require(value, message):
    if not value: raise ValueError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path): return {'path': str(path), 'sha256': sha(path)}
def read(path): return json.loads(Path(path).read_text())
def checked(value):
    require(sha(value['path']) == value['sha256'], 'Input changed: ' + value['path'])
    return read(value['path'])


def fixed_route(ticket, capacity, routes, *, now=None):
    """Use only the exact prepared node/account/type; no fallback route."""
    now = time.time() if now is None else now
    require(capacity.get('status') == 'observed', 'Fresh preparation capacity unavailable')
    _, suffix, values, demand = routes.sbatch_parts(ticket['sbatch_argv'])
    require(values.get('--nodelist') == 'svl7' and values.get('--gres') == 'gpu:titanrtx:1', 'Recovery must stay pinned to svl7 Titan')
    require(values.get('--account') == 'vision' and ticket['expected_account'] == 'minyeong', 'Recovery account differs')
    fresh = routes.build_capacity(capacity['probe'], demand, now=now, reservations=capacity.get('_reservations', []))
    require(fresh['critical_titan_reserve_policy'] == 4, 'Four-Titan reserve differs')
    nodes = routes.inventory(capacity['probe'], demand, capacity.get('_reservations', []), values.get('--exclude', '').split(','))
    _, held = routes.reserve_titans(nodes)
    node = nodes.get('svl7')
    require(held == 4 and node is not None and routes.node_slots(node, 'titanrtx', demand) >= 1,
            'svl7 unavailable after pending work, CPU/memory and four-Titan reserve')
    allowed = [r for r in fresh['routes'] if r['user'] == 'minyeong' and r['slurm_account'] == values['--account']
               and r['partition'] in values['--partition'].split(',') and r['partition'] in node['partitions']
               and r['slots'] > 0 and r['model_slots'].get('titanrtx', 0) > 0]
    require(allowed, 'svl7 exact account/partition quota unavailable')
    route = allowed[0]
    return {'command': ticket['sbatch_argv'], 'prepared_command': ticket['sbatch_argv'],
            'payload_sha256': routes.digest(suffix), 'capacity_observed_at': fresh['observed_at'],
            'query_sha256': capacity['probe']['query_sha256'], 'critical_titan_slots_actually_held': held,
            'route': {'node': 'svl7', 'gpu_model': 'titanrtx', 'partition': route['partition'],
                      'slurm_account': 'vision', 'user': 'minyeong', **demand}}


def verify_prepared_recovery(prepared, expected):
    recovery = prepared['recovery']
    require(recovery.get('proof', recovery) == expected, 'Prepared recovery proof differs')
    require(recovery.get('maximum_attempts', 1) == 1, 'Prepared attempt bound differs')


def verify_request(request_ref):
    request = checked(request_ref); manifest = checked(request['ticket']); proof = checked(request['recovery_proof'])
    require(request['schema'] == 'isg-root-reviewed-single-recovery-v1' and len(manifest['jobs']) == 1, 'Unsupported recovery')
    ticket = manifest['jobs'][0]; key = request['physical_dedup_key']; directory = Path(request['ticket']['path']).parent
    require(key == ticket['physical_dedup_key'] == ticket['dedup_key'] == proof['physical_dedup_key'], 'Physical identity differs')
    require(proof['maximum_attempts'] == 1 and ticket['native_time_seconds'] == 1500 and ticket['hard_time_seconds'] == 1800, 'Recovery bound differs')
    for item in ticket['input_refs'] + ticket['scripts'] + [ticket['runtime_ref'], ticket['prepared']]:
        require(sha(item['path']) == item['sha256'], 'Recovery input changed: ' + item['path'])
    prepared = checked(ticket['prepared'])
    identity = hashlib.sha256(json.dumps(prepared['physical_identity'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(identity == key, 'Prepared identity differs')
    verify_prepared_recovery(prepared, request['recovery_proof'])
    terminal = checked(proof['previous_terminal']); checked(proof['previous_submission']); checked(proof['previous_claim'])
    require(terminal['physical_dedup_key'] == key and not terminal['map_files'] and not any(a['native_done'] for a in terminal['arms']), 'Prior output is not the exact startup failure')
    require(sha(proof['previous_native_log']['path']) == proof['previous_native_log']['sha256'], 'Prior native log changed')
    require("ModuleNotFoundError: No module named 'omni.kit.usd'" in Path(proof['previous_native_log']['path']).read_text(), 'Startup diagnosis differs')
    script = next(r for r in ticket['scripts'] if Path(r['path']).name == 'job.sh')
    expected = 'export KIT_TPL_NFS_ROOT=' + str(directory / 'private_kit_templates')
    require(expected in Path(script['path']).read_text().splitlines(), 'Private Kit template root differs')
    require(ticket['claim_path'] == request['original_claim_path'] and ticket['submission_path'] == request['original_submission_path'], 'Original claim lineage differs')
    return request, ticket, proof, terminal


def unchanged_original(request, proof):
    require(sha(request['original_claim_path']) == proof['previous_claim']['sha256'], 'Original physical claim changed')
    require(sha(request['original_submission_path']) == proof['previous_submission']['sha256'], 'Global receipt changed; reconcile existing attempt')
    checked(proof['previous_terminal'])


def submit_once(request_ref, root, routes, *, execute=False, runner=subprocess.run, probe=None, writer=None):
    if writer is None:
        from scripts.campaign_20261110.infra_dispatch import shared_json as writer
    request, ticket, proof, terminal = verify_request(request_ref)
    directory = Path(request['ticket']['path']).parent; intent = directory / 'attempt_intent.json'
    receipt = Path(request['root_attempt_receipt_path'])
    if intent.exists() or receipt.exists(): return {'status': 'existing_attempt_requires_reconciliation', 'intent': str(intent), 'receipt': str(receipt)}
    unchanged_original(request, proof)
    result = {'request': request_ref, 'physical_dedup_key': ticket['physical_dedup_key'], 'preflight_at': time.time(), 'status': 'preflight_passed', 'submitted': False}
    if not execute: return result
    with routes.allocation_lock(root):
        require(not intent.exists() and not receipt.exists(), 'Concurrent recovery intent exists')
        unchanged_original(request, proof)
        observation = runner(['sacct', '-X', '-n', '-P', '-j', str(terminal['job_id']), '--format=JobIDRaw,State,ExitCode'], capture_output=True, text=True, timeout=15, check=True)
        states = [line.split('|') for line in observation.stdout.splitlines() if line.strip()]
        require(any(row[0] == str(terminal['job_id']) and row[1].split()[0].rstrip('+') in {'FAILED', 'COMPLETED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY', 'CANCELLED'} for row in states), 'Original job is not freshly terminal')
        capacity = (probe or routes.probe)()
        if capacity.get('status') == 'observed': routes.reserve_unobserved_receipts(root, capacity)
        try: routing = fixed_route(ticket, capacity, routes)
        except ValueError as error:
            result.update(status='capacity_blocked', blocker=str(error), capacity=capacity, observed_at=time.time())
            writer(directory / 'capacity_blocker.json', result); return result
        record = {**ticket, 'command': ticket['sbatch_argv'], 'preparation_route': routing, 'status': 'submitting',
                  'job_id': None, 'backend': 'cluster', 'observed_at': time.time(), 'previous_submission': proof['previous_submission'],
                  'previous_job_id': str(terminal['job_id']), 'recovery_request': request_ref, 'dispatch_owner': '/root', 'maximum_attempts': 1}
        with intent.open('x') as stream: json.dump(record, stream, indent=2)
        writer(request['original_submission_path'], record)
        # Any interruption from this point preserves intent and capacity reservation.
        try:
            run = runner(ticket['sbatch_argv'], capture_output=True, text=True, timeout=60)
            jid = run.stdout.strip().split(';')[0]
            good = run.returncode == 0 and jid.isdigit()
            record.update(returncode=run.returncode, stderr=run.stderr, stdout=run.stdout, job_id=jid if good else None,
                          status='submitted' if good else 'submission_requires_reconciliation', submitted_at=time.time())
        except subprocess.SubprocessError as error:
            record.update(status='submission_requires_reconciliation', error=repr(error), observed_at=time.time())
        writer(receipt, record); writer(request['original_submission_path'], record)
        require(sha(request['original_claim_path']) == proof['previous_claim']['sha256'], 'Physical claim changed unexpectedly')
        return {'status': record['status'], 'job_id': record.get('job_id'), 'receipt': ref(receipt), 'submitted': record['status'] == 'submitted'}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--manifest', required=True); parser.add_argument('--sha256', required=True); parser.add_argument('--execute', action='store_true'); args = parser.parse_args()
    manifest = checked({'path': args.manifest, 'sha256': args.sha256}); require(len(manifest['requests']) == 2 and manifest['owner'] == '/root', 'Exactly two root-reviewed recoveries required')
    for item in manifest['helper_inputs']: require(sha(item['path']) == item['sha256'], 'Pinned capacity helper changed')
    helper = manifest['route_helper']; spec = importlib.util.spec_from_file_location('reviewed_fixed_recovery_routes', helper['path']); routes = importlib.util.module_from_spec(spec); spec.loader.exec_module(routes)
    results = [submit_once(item, Path(manifest['root']), routes, execute=args.execute) for item in manifest['requests']]
    print(json.dumps({'observed_at': time.time(), 'results': results, 'owner': '/root', 'execute': args.execute}), flush=True)


if __name__ == '__main__': main()
