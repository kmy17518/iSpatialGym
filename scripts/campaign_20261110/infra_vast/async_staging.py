"""Durable reservations around bounded staging subprocesses; no GPU assignment here."""
import concurrent.futures
import json
import os
from pathlib import Path
import time
import uuid


class StagingReservations:
    active_statuses = {'preparing', 'ready', 'assigning'}

    def __init__(self, root, owner, max_workers=4):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.owner = str(owner)
        self.limit = max_workers
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.futures = {}
        self.records = {p.stem:json.loads(p.read_text()) for p in self.root.glob('*.json')}

    def save(self, record):
        path = self.root / (record['id'] + '.json')
        temporary = path.with_suffix('.' + uuid.uuid4().hex + '.tmp')
        with temporary.open('x') as stream:
            json.dump(record,stream,indent=2);stream.flush();os.fsync(stream.fileno())
        temporary.replace(path);self.records[record['id']] = record

    def archive(self, record, reason):
        directory = self.root/'history';directory.mkdir(exist_ok=True)
        path = directory/(record['id']+'_'+uuid.uuid4().hex+'.json')
        with path.open('x') as stream: json.dump({**record,'archived_reason':reason},stream,indent=2)
        return str(path)

    def active(self):
        return {jid:row for jid,row in self.records.items() if row['status'] in self.active_statuses}

    def reservations(self):
        result = {}
        for jid,row in self.active().items():result.setdefault(row['instance'],[]).append(jid)
        return result

    def recover(self, owner_alive, remote_assignment, reconcile):
        """An interrupted upload is retried only after its previous owner is dead.

        Assignment has a write-ahead record. A crash after R2 publication is
        reconciled to that exact worker before the reservation can be reused.
        """
        for jid,row in list(self.active().items()):
            if jid in self.futures:continue
            if row['owner'] != self.owner and owner_alive(row['owner']):continue
            if row['status'] == 'preparing':
                prior = self.archive(row,'coordinator_stopped_before_stage_completion')
                self.save({**row,'status':'interrupted','previous':prior,'recovered_at':time.time()})
            elif row['status'] == 'assigning':
                assigned = remote_assignment(row['instance'],jid)
                if assigned is not None:
                    if assigned != row['spec']:raise ValueError('remote assignment differs from durable staging reservation: '+jid)
                    reconcile(row)
                    self.save({**row,'status':'assigned','reconciled_at':time.time()})
                # An absent object after an uncertain PUT never frees the job
                # for another worker. The caller may retry the identical PUT
                # to this worker; late completion is then idempotent.
            elif row['owner'] != self.owner:
                self.save({**row,'owner':self.owner,'recovered_at':time.time()})

    def submit(self, job, instance, worker, config, function):
        jid = str(job['id'])
        if jid in self.active():raise ValueError('job already has a staging reservation')
        if len(self.futures) >= self.limit:return False
        prior = self.records.get(jid)
        record = {'id':jid,'owner':self.owner,'status':'preparing','instance':str(instance),
                  'worker':dict(worker),'job':dict(job),'started_at':time.time(),
                  'timeout_seconds':min(900,max(30,int(config.get('stage_timeout_s',600))))}
        if prior:record['previous'] = self.archive(prior,'fresh_staging_attempt')
        self.save(record)  # reservation is durable before any subprocess starts
        self.futures[jid] = self.pool.submit(function,dict(job),dict(config))
        return True

    def poll(self):
        failed = []
        for jid,future in list(self.futures.items()):
            if not future.done():continue
            row=self.records[jid]
            try:
                spec=future.result()
                self.save({**row,'status':'ready','spec':spec,'stage_completed_at':time.time()})
            except Exception as error:
                row={**row,'status':'failed','error':repr(error),'failed_at':time.time()}
                self.save(row);failed.append(row)
            del self.futures[jid]
        return failed

    def ready(self):
        return [row for row in self.active().values() if row['status']=='ready']

    def assigning(self, jid, spec):
        row=self.records[jid]
        if row['status']!='ready':raise ValueError('stage is not complete')
        self.save({**row,'status':'assigning','spec':spec,'assignment_started_at':time.time()})

    def assigned(self,jid):
        self.save({**self.records[jid],'status':'assigned','assigned_at':time.time()})

    def release(self,jid,reason):
        row=self.records[jid]
        if row['status']=='assigning':raise ValueError('ambiguous assignment must be reconciled first')
        self.save({**row,'status':'released','release_reason':reason,'released_at':time.time()})
