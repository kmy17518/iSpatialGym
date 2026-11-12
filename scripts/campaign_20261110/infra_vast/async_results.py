"""Bounded result I/O reservations; only the coordinator commits job state.

Workers retain the original GPU assignment until successful result import. A
failed/interrupted import is preserved for reconciliation, never a GPU retry.
"""
import concurrent.futures
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

KINDS = {'tabletop', 'attachment', 'tabletop_maps', 'contact_maps'}
NAMES = {'isg261110_library_prebuild', 'isg261110_map_prebuild', 'isg261110_contact_map'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def typed_preparation(job):
    if job.get('job_kind') != 'native_dependency' or job.get('name') not in NAMES:
        return False
    path = Path(job['job_dir'])/'native_inputs.json'
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != job.get('native_inputs_sha256'):
        raise ValueError('Result fetch native-input identity differs')
    value = json.loads(payload)
    return (value.get('kind') in KINDS and value.get('new_native_instances') == 0
            and value.get('production_released') is False and value.get('scenes') == [])


class CoordinatorStateCache:
    """Only a sole coordinator may retain this mapping between ticks.

    A failed durable write or interrupted tick invalidates the mapping; the
    next access reloads disk. Return callbacks share the exact owned objects.
    """
    def __init__(self, loader):
        self.loader = loader; self.value = None; self.dirty = True

    def get(self):
        if self.dirty:
            self.value = self.loader(); self.dirty = False
        return self.value

    def invalidate(self):self.dirty = True

    def require_durable(self):
        if self.dirty:raise RuntimeError('Job state persistence failed; reload before further side effects')


def identity(job, done):
    # Keep assignment identity separate from status/sync_error updated by main.
    return digest({'job': {key: job.get(key) for key in
        ('id', 'job_dir', 'log', 'instance', 'assigned', 'tries', 'native_inputs_sha256')}, 'done': done})


@contextmanager
def output_locks(root, row, *, nonblocking=False):
    """Kernel locks also cover a child surviving its former coordinator."""
    root = Path(root)/'locks'; root.mkdir(exist_ok=True); streams = []
    keys = ['job_'+row['id'], 'directory_'+hashlib.sha256(os.path.realpath(row['job']['job_dir']).encode()).hexdigest()]
    try:
        for key in sorted(keys):
            stream = (root/(key+'.lock')).open('a'); streams.append(stream)
            fcntl.flock(stream, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        yield
    finally:
        for stream in reversed(streams):stream.close()


class ResultFetchReservations:
    active_statuses = {'fetching', 'ready'}

    def __init__(self, root, owner, max_workers=8):
        if type(max_workers) is not int or not 1 <= max_workers <= 8:
            raise ValueError('Result fetch concurrency must be 1..8')
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.owner = str(owner); self.limit = max_workers
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.futures = {}; self.records = {p.stem: json.loads(p.read_text()) for p in self.root.glob('*.json')}
        self.reconciled = set()

    def close(self):self.pool.shutdown(wait=True)

    def save(self, row):
        write(self.root/(row['id']+'.json'), row); self.records[row['id']] = row

    def metrics(self):
        return {'parallelism': self.limit, 'in_flight': len(self.futures),
                'ready': sum(r['status'] == 'ready' for r in self.records.values()),
                'failed_or_interrupted': sum(r['status'] in ('failed', 'interrupted') for r in self.records.values())}

    def recover(self, owner_alive):
        live_owners = {}
        for jid, row in list(self.records.items()):
            if jid in self.futures or row['status'] not in self.active_statuses:continue
            if jid in self.reconciled:continue
            if row['owner'] != self.owner:
                if row['owner'] not in live_owners:live_owners[row['owner']] = owner_alive(row['owner'])
                if live_owners[row['owner']]:continue
            try:
                with output_locks(self.root, row, nonblocking=True):
                    result = Path(row['result'])
                    if result.exists():
                        value = json.loads(result.read_text()); self._verify(row, value)
                        self.save({**row, 'owner': self.owner, 'status': 'ready', 'value': value['value'], 'recovered_at': time.time()})
                    elif row['status'] == 'fetching':
                        self.save({**row, 'status': 'interrupted', 'error': 'Former fetch ended without a verified completion; partials retained', 'recovered_at': time.time()})
                    else:
                        self.save({**row, 'owner': self.owner})
                    self.reconciled.add(jid)
            except BlockingIOError:
                continue
            except ValueError as error:
                self.save({**row, 'owner': self.owner, 'status': 'failed',
                           'error': repr(error), 'failed_at': time.time()})

    @staticmethod
    def _verify(row, value):
        if value.get('token') != row['token'] or value.get('identity') != row['identity']:
            raise ValueError('Result completion belongs to another assignment')
        if type(value.get('value')) is not int or value['value'] < 0:
            raise ValueError('Invalid result cache count')

    def submit(self, job, done, function, timeout=180):
        jid = str(job['id'])
        if not jid.isdigit():raise ValueError('Unsafe result job identifier')
        if jid in self.records:return False  # includes failures: no implicit retry
        occupied = [r for r in self.records.values() if r['status'] in self.active_statuses]
        if len(occupied) >= self.limit:return False
        directory = os.path.realpath(job['job_dir'])
        if any(os.path.realpath(r['job']['job_dir']) == directory for r in occupied):return False
        token = uuid.uuid4().hex; row = {'id': jid, 'owner': self.owner, 'status': 'fetching',
            'job': dict(job), 'done': dict(done), 'identity': identity(job, done), 'token': token,
            'started_at': time.time(), 'timeout_seconds': min(600, max(30, int(timeout))),
            'result': str(self.root/'completed'/(jid+'_'+token+'.json'))}
        self.save(row)  # Durable ownership precedes subprocess creation.
        self.futures[jid] = self.pool.submit(function, dict(row), str(self.root/(jid+'.json')))
        return True

    def poll(self):
        for jid, future in list(self.futures.items()):
            if not future.done():continue
            row = self.records[jid]
            try:
                value = future.result(); self._verify(row, value)
                self.save({**row, 'status': 'ready', 'value': value['value'], 'completed_at': time.time()})
                self.reconciled.add(jid)
            except Exception as error:
                self.save({**row, 'status': 'failed', 'error': repr(error), 'failed_at': time.time()})
            del self.futures[jid]

    def result(self, job, done):
        row = self.records.get(str(job['id']))
        if row is None:return None
        if row['identity'] != identity(job, done):raise ValueError('Result reservation assignment changed')
        if row['owner'] != self.owner:return None
        if row['status'] in ('failed', 'interrupted'):raise RuntimeError(row['error'])
        return row['value'] if row['status'] == 'ready' else None

    def committed(self, jid):
        row = self.records[str(jid)]
        if row['status'] != 'ready' or row['owner'] != self.owner:raise ValueError('Fetch is not owned and ready')
        self.save({**row, 'status': 'committed', 'committed_at': time.time()})

    def retains_assignment(self, job):
        row = self.records.get(str(job['id']))
        return bool(row and row['status'] != 'committed'
                    and identity(job, row['done']) == row['identity'])

    def reconcile_committed(self, states):
        for jid, row in list(self.records.items()):
            job = states.get(jid, {})
            if (row['owner'] == self.owner and row['status'] == 'ready'
                    and job.get('status') in ('synced', 'failed') and job.get('done') == row['done']
                    and identity(job, row['done']) == row['identity']):
                self.committed(jid)


def execute_reserved_fetch(path, job, fetch):
    """Child entry: no job-state/ledger/acceptance writes are available here."""
    path = Path(path); row = json.loads(path.read_text())
    if row['status'] != 'fetching' or identity(job, row['done']) != row['identity']:
        raise ValueError('Result child assignment differs')
    if job.get('status') != 'assigned' or not typed_preparation(job):
        raise ValueError('Result child requires assigned typed preparation')
    with output_locks(path.parent, row, nonblocking=True):
        current = json.loads(path.read_text())
        if any(current.get(key) != row.get(key) for key in ('status', 'owner', 'token', 'identity')):
            raise ValueError('Result reservation changed before child acquired output locks')
        added = fetch(job)
        value = {'token': row['token'], 'identity': row['identity'], 'value': added,
                 'completed_at': time.time()}
        ResultFetchReservations._verify(row, value); write(row['result'], value)
    return value
