import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.campaign_20261110 import controller_held as held
from scripts.campaign_20261110.storage import atomic_directory_lock


class PlanningLockTests(unittest.TestCase):
    def test_retry_after_live_owner_release(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'planner.lockdir'
            owner=atomic_directory_lock(path);owner.__enter__()
            original=(path/'owner.json').read_bytes()
            def release(_):
                self.assertEqual((path/'owner.json').read_bytes(),original)
                owner.__exit__(None,None,None)
            with patch.object(held.time,'sleep',side_effect=release) as sleep:
                with held.planning_lock(path):
                    self.assertTrue(path.exists())
                self.assertEqual(sleep.call_count,1)
            self.assertFalse(path.exists())

    def test_expiry_keeps_other_owner_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'planner.lockdir'
            with atomic_directory_lock(path):
                before=(path/'owner.json').read_bytes()
                with self.assertRaises(FileExistsError):
                    with held.planning_lock(path,timeout=0):self.fail('stole the lock')
                self.assertEqual((path/'owner.json').read_bytes(),before)

    def test_transaction_exception_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'planner.lockdir';calls=[]
            with patch.object(held.time,'sleep') as sleep:
                with self.assertRaisesRegex(FileExistsError,'transaction side effect'):
                    with held.planning_lock(path):
                        calls.append('authored')
                        raise FileExistsError('transaction side effect')
                self.assertEqual(calls,['authored']);sleep.assert_not_called()
                self.assertFalse(path.exists())


if __name__=='__main__':unittest.main()
