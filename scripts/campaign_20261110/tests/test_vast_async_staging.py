import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from scripts.campaign_20261110.infra_vast.async_staging import StagingReservations, staging_parallelism, staging_compression_threads


class AsyncStagingTests(unittest.TestCase):
    def manager(self, root, owner='100'):
        manager=StagingReservations(root,owner,2)
        self.addCleanup(manager.close)
        return manager

    def test_slow_stage_does_not_block_other_completed_stage_or_duplicate_reservation(self):
        with tempfile.TemporaryDirectory() as root:
            gate=threading.Event();self.addCleanup(gate.set)
            manager=self.manager(root)
            def stage(job,config):
                if job['id']=='1':gate.wait(2)
                return {'inputs':{},'id':job['id']}
            self.assertTrue(manager.submit({'id':'1'},'w1',{}, {},stage))
            self.assertTrue(manager.submit({'id':'2'},'w2',{}, {},stage))
            with self.assertRaises(ValueError):manager.submit({'id':'1'},'w3',{}, {},stage)
            deadline=time.time()+1
            while not manager.ready() and time.time()<deadline:manager.poll();time.sleep(.01)
            self.assertEqual([x['id'] for x in manager.ready()],['2'])
            self.assertEqual(manager.reservations(),{'w1':['1'],'w2':['2']})
            self.assertEqual(json.loads((Path(root)/'1.json').read_text())['status'],'preparing')
            gate.set()

    def test_restart_never_reuses_live_owner_reservation(self):
        with tempfile.TemporaryDirectory() as root:
            manager=self.manager(root)
            manager.save({'id':'1','owner':'99','status':'preparing','instance':'w1'})
            manager.recover(lambda owner:True,lambda *args:None,lambda row:None)
            self.assertIn('1',manager.active())
            manager.recover(lambda owner:False,lambda *args:None,lambda row:None)
            self.assertNotIn('1',manager.active())
            self.assertTrue(Path(manager.records['1']['previous']).is_file())

    def test_restart_after_assignment_publish_reconciles_same_worker_once(self):
        with tempfile.TemporaryDirectory() as root:
            manager=self.manager(root);spec={'id':'1','inputs':{'source':'sha'}}
            manager.save({'id':'1','owner':'99','status':'assigning','instance':'w1','spec':spec})
            reconciled=[]
            manager.recover(lambda owner:False,lambda worker,jid:spec,reconciled.append)
            manager.recover(lambda owner:False,lambda worker,jid:spec,reconciled.append)
            self.assertEqual(len(reconciled),1);self.assertEqual(reconciled[0]['instance'],'w1')
            self.assertEqual(manager.records['1']['status'],'assigned')

    def test_assignment_mismatch_cannot_release_or_reassign(self):
        with tempfile.TemporaryDirectory() as root:
            manager=self.manager(root)
            manager.save({'id':'1','owner':'99','status':'assigning','instance':'w1','spec':{'id':'1'}})
            with self.assertRaisesRegex(ValueError,'differs'):
                manager.recover(lambda owner:False,lambda *args:{'id':'other'},lambda row:None)
            with self.assertRaisesRegex(ValueError,'ambiguous'):manager.release('1','retry')
            self.assertIn('1',manager.active())
            manager.recover(lambda owner:False,lambda *args:None,lambda row:None)
            self.assertEqual(manager.records['1']['status'],'assigning')
            self.assertEqual(manager.reservations(),{'w1':['1']})

    def test_failure_keeps_record_and_releases_slot(self):
        with tempfile.TemporaryDirectory() as root:
            manager=self.manager(root)
            def fail(*args):raise RuntimeError('bounded stage timed out; preserved log')
            manager.submit({'id':'1'},'w1',{}, {},fail)
            errors=[];deadline=time.time()+1
            while not errors and time.time()<deadline:errors=manager.poll();time.sleep(.01)
            self.assertEqual(len(errors),1);self.assertEqual(manager.reservations(),{})
            self.assertIn('preserved log',manager.records['1']['error'])

    def test_reservation_keeps_only_reconciliation_worker_identity(self):
        with tempfile.TemporaryDirectory() as root:
            manager=self.manager(root)
            worker={'id':12,'_key_name':'main','machine_id':34,'dph_total':.5,
                    'gpu_name':'RTX 5090','num_gpus':1,'extra_env':{'PRIVATE_VALUE':'secret'}}
            manager.submit({'id':'1'},'12',worker,{},lambda *args:{'id':'1'})
            saved=json.loads((Path(root)/'1.json').read_text())
            self.assertNotIn('extra_env',saved['worker'])
            self.assertNotIn('secret',json.dumps(saved))
            self.assertEqual(saved['worker'],{key:value for key,value in worker.items() if key!='extra_env'})

    def test_explicit_concurrency_is_not_silently_capped_at_eight(self):
        self.assertEqual(staging_parallelism({'staging_parallelism': 64}), 64)
        self.assertEqual(staging_compression_threads({'stage_compression_threads': 1}), 1)
        for value in (0, -1, True, 3.5, '32'):
            with self.assertRaises(ValueError):
                staging_parallelism({'staging_parallelism': value})
            with self.assertRaises(ValueError):
                staging_compression_threads({'stage_compression_threads': value})

    def test_resize_fills_more_workers_without_losing_inflight_ownership(self):
        with tempfile.TemporaryDirectory() as root:
            manager = self.manager(root)
            gate = threading.Event(); self.addCleanup(gate.set)
            started = set(); lock = threading.Lock()
            def stage(job, config):
                with lock: started.add(job['id'])
                gate.wait(2)
                return {'id': job['id']}
            for n in range(2):
                self.assertTrue(manager.submit({'id': str(n)}, str(n), {}, {}, stage))
            self.assertFalse(manager.submit({'id': '2'}, '2', {}, {}, stage))
            manager.resize(24)
            for n in range(2, 24):
                self.assertTrue(manager.submit({'id': str(n)}, str(n), {}, {}, stage))
            deadline = time.time() + 1
            while len(started) < 24 and time.time() < deadline: time.sleep(.01)
            self.assertEqual(len(started), 24)
            self.assertEqual(len(manager.reservations()), 24)
            self.assertEqual(manager.metrics()['in_flight'], 24)
            manager.resize(1)
            self.assertFalse(manager.submit({'id': '25'}, '25', {}, {}, stage))
            self.assertEqual(len(manager.reservations()), 24)
            self.assertEqual(manager.metrics()['available_submission_slots'], 0)
            gate.set()
            deadline = time.time() + 1
            while manager.futures and time.time() < deadline: manager.poll(); time.sleep(.01)
            self.assertEqual(len(manager.ready()), 24)
            self.assertEqual(len(started), 24)  # no duplicated staging on resize


if __name__=='__main__':unittest.main()
