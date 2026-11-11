"""Allowlisted execution observations from the coordinator's existing reads."""
import hashlib
import json
import math
import os
from pathlib import Path
import uuid


def mirror_heartbeats(directory, heartbeats, observed_at):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for instance, heartbeat in heartbeats.items():
        instance = str(instance)
        if not instance.isdigit() or not isinstance(heartbeat, dict):
            continue
        timestamp = heartbeat.get('time')
        if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool) or not math.isfinite(timestamp):
            continue
        path = directory / (instance + '.json')
        if path.exists():
            prior = json.loads(path.read_text())
            if prior.get('heartbeat', {}).get('time', 0) > timestamp:
                continue
        allowed = ('time', 'phase', 'slots', 'gpu_used_mb', 'load', 'mem_used_gb', 'warm_kit_templates',
                   'exact_ik_exchange_version', 'done')
        selected = {key: heartbeat[key] for key in allowed if key in heartbeat}
        selected['running'] = {}
        for job, state in (heartbeat.get('running') or {}).items():
            if not str(job).isdigit() or not isinstance(state, dict):
                continue
            fields = ('start', 'name', 'cpus', 'phase', 'fetch_started_unix', 'runtime_started_unix',
                      'upload_started_unix')
            selected['running'][str(job)] = {key: state[key] for key in fields if key in state}
        payload = {'schema': 'isg-vast-worker-heartbeat-mirror-v1', 'campaign_id': 'campaign_20261110',
                   'instance_id': instance, 'observed_at_unix': observed_at,
                   'source_object': 'hb/' + instance + '.json',
                   'source_payload_sha256': hashlib.sha256(json.dumps(heartbeat, sort_keys=True).encode()).hexdigest(),
                   'heartbeat': selected,
                   'boundary': 'Worker-reported execution observation; compare original heartbeat time and exact job ID. Not acceptance.'}
        temporary = path.with_suffix('.' + uuid.uuid4().hex + '.tmp')
        try:
            temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + '\n')
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        written.append(str(path))
    return written
