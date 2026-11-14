"""Request/reply transport for online workers without the cluster filesystem.

The broker serializes goal claims using OnlineGroups.GoalRegistry. A worker
waits inside its loaded sampling loop; no discovery pass or member dispatch is
introduced. Requests and responses stay within the production fleet R2 prefix.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from scripts.campaign_20261110.contracts import digest, require, write_json_atomic

PREFIX = 's3://isg-demos/jobs/og-isg261110/online-goal-registry/'


class Mailbox:
    def __init__(self, location):
        self.location = location.rstrip('/')
        self.remote = self.location.startswith('s3://')
        if self.remote:
            require((self.location + '/').startswith(PREFIX), 'registry is outside production R2 prefix')
            self.aws = ['aws']
            endpoint = os.environ.get('AWS_ENDPOINT_URL') or os.environ.get('R2_ENDPOINT_URL')
            if endpoint: self.aws += ['--endpoint-url', endpoint]
        else:
            Path(self.location).mkdir(parents=True, exist_ok=True)

    def get(self, key):
        if not self.remote:
            p = Path(self.location) / key
            return json.loads(p.read_text()) if p.exists() else None
        result = subprocess.run(self.aws + ['s3', 'cp', self.location + '/' + key, '-'], capture_output=True, timeout=20)
        if result.returncode:
            if b'404' in result.stderr or b'NoSuchKey' in result.stderr: return None
            raise RuntimeError('registry object read failed; credentials/network must be checked')
        return json.loads(result.stdout)

    def put(self, key, value):
        if not self.remote:
            write_json_atomic(Path(self.location) / key, value); return
        result = subprocess.run(self.aws + ['s3', 'cp', '-', self.location + '/' + key, '--only-show-errors'],
                                input=json.dumps(value).encode(), capture_output=True, timeout=20)
        require(result.returncode == 0, 'registry object write failed')

    def requests(self):
        if not self.remote:
            return [str(p.relative_to(self.location)) for p in (Path(self.location) / 'requests').glob('*.json')]
        result = subprocess.run(self.aws + ['s3', 'ls', self.location + '/requests/'], capture_output=True, timeout=20)
        require(result.returncode == 0, 'registry listing failed')
        return ['requests/' + line.split()[-1] for line in result.stdout.decode().splitlines() if line.split()[-1].endswith('.json')]


def mailbox_claim(location, record, owner, deadline):
    box = Mailbox(location)
    request = {'record': record, 'owner': owner, 'deadline': deadline}
    key = digest({'record': record, 'owner': owner})
    box.put('requests/' + key + '.json', request)
    while time.time() < deadline:
        reply = box.get('responses/' + key + '.json')
        if reply is not None:
            require(reply['request_sha256'] == digest(request), 'registry response request mismatch')
            require(not reply.get('error'), 'registry refused request: ' + str(reply.get('error')))
            return reply['claim']
        time.sleep(min(1., max(0., deadline - time.time())))
    raise TimeoutError('online goal claim deadline')


def serve_once(location, registry_root):
    from ispatialgym.sampling.online_groups import GoalRegistry
    box = Mailbox(location); registry = GoalRegistry(registry_root)
    for key in box.requests():
        target = key.replace('requests/', 'responses/', 1)
        if box.get(target) is not None: continue
        request = box.get(key)
        reply = {'request_sha256': digest(request)}
        try:
            require(request['owner'].startswith('og-isg261110-'), 'unknown production worker')
            require(time.time() < request['deadline'], 'expired goal claim request')
            reply['claim'] = registry.claim(request['record'], request['owner'], min(request['deadline'], time.time() + 5))
        except (ValueError, TimeoutError) as exc:
            reply['error'] = str(exc)
        box.put(target, reply)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mailbox', required=True); parser.add_argument('--registry', required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    while True:
        serve_once(args.mailbox, args.registry)
        if args.once: break
        time.sleep(1.)
