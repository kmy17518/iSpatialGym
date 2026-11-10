import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from scripts.campaign_20261110.infra_vast.async_staging import StagingReservations


class AsyncStagingTests(unittest.TestCase):
    def manager(self, root, owner='100'):
        manager=StagingReservations(root,owner,2)
        self.addCleanup(manager.pool.shutdown,wait=True)
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


if __name__=='__main__':unittest.main()
