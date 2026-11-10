"""Pure per-contract resource checks shared by coordinator and worker."""
import json


def bad_machines(events):
    """Account retirement is not evidence of a host failure.

    Older coordinator code matched the word 'heartbeat' in this exact drain
    description. Preserve those ledger events but correct their classification.
    """
    misclassified = 'account funding drain; no assigned/staging or heartbeat-running jobs'
    return {event.get('machine_id') for event in events
            if event.get('ev') == 'bad_machine' and event.get('reason') != misclassified}


def host_startup_failure(reason):
    return reason.startswith(('no heartbeat after ', 'boot not finished after '))


def gpu_ids(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            value = value.split(',')
    return {str(x).strip() for x in (value or []) if str(x).strip()}


def machine_reservation(offer, live, heartbeats, created, config):
    """Reserve distinct physical GPUs and CPUs before adding a host contract."""
    machine = offer.get('machine_id')
    if machine is None:
        return None, 'missing_machine_identity'
    peers = [(str(iid), row) for iid, row in live.items() if row.get('machine_id') == machine]
    if len(peers) >= int(config.get('max_contracts_per_machine', 4)):
        return None, 'machine_contract_cap'
    ids = gpu_ids(offer.get('gpu_ids'))
    if len(ids) != 1 or int(offer.get('num_gpus') or 0) != 1:
        return None, 'single_gpu_identity_required'
    needed = int(config['cpus_per_job']) + 1
    if float(offer.get('cpu_cores_effective') or 0) < needed or float(offer.get('cpu_ram') or 0) / 1024 < config['ram_per_job_gb']:
        return None, 'insufficient_guaranteed_resources'
    occupied, allowed = set(), set()
    for iid, row in peers:
        prior, hb = created.get(iid, {}), heartbeats.get(iid, {})
        previous_ids = gpu_ids(row.get('gpu_ids') or prior.get('gpu_ids'))
        if not previous_ids:
            return None, 'peer_gpu_identity_unknown'
        if ids & previous_ids:
            return None, 'duplicate_physical_gpu'
        cpus = hb.get('selected_cpu_affinity') or prior.get('cpu_affinity')
        affinity = hb.get('allowed_cpu_affinity') or prior.get('machine_cpu_affinity')
        if not cpus and affinity and hb.get('cpus'):
            cpus = sorted(affinity)[:int(hb['cpus'])]  # unchanged legacy worker rule
        if not cpus or not affinity:
            return None, 'peer_cpu_reservation_unknown'
        occupied.update(cpus)
        allowed.update(affinity)
    if not peers:
        return {'gpu_ids': sorted(ids), 'cpu_affinity': [], 'machine_cpu_affinity': []}, None
    free = sorted(allowed - occupied)
    if len(free) < needed:
        return None, 'insufficient_disjoint_cpus'
    return {'gpu_ids': sorted(ids), 'cpu_affinity': free[:needed], 'machine_cpu_affinity': sorted(allowed)}, None


def select_cpus(affinity, quota_count, requested):
    allowed = sorted(set(affinity))
    limit = len(allowed) if quota_count is None else max(0, int(quota_count))
    if requested:
        requested = sorted(set(int(x) for x in requested))
        if not set(requested).issubset(allowed) or len(requested) > limit:
            return []
        return requested
    return allowed[:limit]


def duplicate_worker_gpus(live, heartbeats):
    """New UUID telemetry supplements provider physical GPU contract identities."""
    seen, blocked = {}, set()
    for iid in sorted(live):
        hb = heartbeats.get(iid, {})
        if hb.get('n_gpus') not in (None, 1):
            blocked.add(iid)
        uuid = hb.get('gpu_uuid')
        key = (live[iid].get('machine_id'), uuid)
        if uuid and key in seen:
            blocked.update((iid, seen[key]))
        elif uuid:
            seen[key] = iid
    return blocked
