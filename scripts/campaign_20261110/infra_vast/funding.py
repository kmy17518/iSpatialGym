"""Read-only account credit evidence and conservative future-lease budgets."""
import concurrent.futures
import json
import math
import os
from pathlib import Path
import time
import uuid


def remaining_credit(snapshot, account, events, timestamp, *, reserve=25.0, maximum_age=600):
    row = snapshot.get('accounts', {}).get(account, {})
    observed = float(row.get('observed_at_unix') or 0)
    credit = row.get('credit_usd')
    if not isinstance(credit, (int, float)) or not math.isfinite(credit) or not 0 <= timestamp - observed <= maximum_age:
        return 0.0
    creations, endings = {}, {}
    for event in events:
        instance = str(event.get('instance'))
        if event['ev'] == 'create' and event.get('account') == account:
            creations[instance] = event
        elif event['ev'] == 'destroy' and event.get('ok'):
            endings.setdefault(instance, float(event['time']))
    spent_since_observation = 0.0
    committed = 0.0
    for instance, event in creations.items():
        start = float(event['time']); rate = float(event['dph_total']) / 3600
        end = min(timestamp, endings.get(instance, timestamp))
        spent_since_observation += rate * max(0, end - max(start, observed))
        if instance not in endings:
            committed += rate * max(0, start + float(event.get('max_lifetime_s', 21600)) - timestamp)
    return max(0.0, float(credit) - reserve - spent_since_observation - committed)


def read_snapshot(path, account_names, *, refresh_seconds=300):
    """Never expose or persist account responses, API keys, or payment details."""
    import requests
    path = Path(path); timestamp = time.time()
    previous = json.loads(path.read_text()) if path.exists() else {}
    if timestamp - float(previous.get('checked_at_unix') or 0) < refresh_seconds:
        return previous

    def probe(account):
        try:
            response = requests.get('https://console.vast.ai/api/v0/users/current/',
                                    headers={'Authorization': 'Bearer ' + os.environ[account]}, timeout=20)
            response.raise_for_status(); body = response.json(); body = body.get('user', body)
            credit = body.get('credit')
            if not isinstance(credit, (int, float)) or not math.isfinite(credit):
                raise ValueError('provider omitted numeric credit')
            return account, {'credit_usd': float(credit), 'observed_at_unix': time.time()}
        except Exception as error:
            old = previous.get('accounts', {}).get(account, {})
            return account, {**old, 'last_error_type': type(error).__name__, 'last_error_at_unix': time.time()}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        accounts = dict(executor.map(probe, account_names))
    result = {'schema': 'isg-vast-account-credit-v1', 'checked_at_unix': timestamp, 'accounts': accounts,
              'scope': 'read-only numeric provider credit; no billing changes'}
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    with temporary.open('x') as stream:
        json.dump(result, stream, indent=2); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
    return result
