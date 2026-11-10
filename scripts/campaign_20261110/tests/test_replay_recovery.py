import json
from pathlib import Path
import tempfile
import unittest
from scripts.campaign_20261110.controller_validation import apply_retry_request, publish_recovered_gate

class ReplayRecoveryTests(unittest.TestCase):
    def fixture(self, root):
        folder=root/'validation/open_loop/class/123'
        folder.mkdir(parents=True)
        (folder/'original_failure.log').write_text('original failure')
        entry={'source_job_id':'123','key':'class','status':'replay_missing','replay_job_id':'456','output':str(folder)}
        request={'request_id':'02','expected_replay_job_id':'456','reason':'audited native room-filter preflight repair'}
        return folder,entry,request

    def test_retry_preserves_old_evidence_and_publishes_only_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder,entry,request=self.fixture(root)
            self.assertTrue(apply_retry_request(entry,request,root,{}))
            self.assertEqual(entry['attempt_history'][0]['replay_job_id'],'456')
            out=Path(entry.pop('retry_output'));out.mkdir()
            entry.update(output=str(out),replay_job_id='789',gate={'success':False})
            (out/'gate.json').write_text('{"success":false}')
            publish_recovered_gate(entry,root)
            self.assertFalse((folder/'gate.json').exists())
            entry['gate']={'success':True};(out/'gate.json').write_text('{"success":true}')
            publish_recovered_gate(entry,root)
            self.assertEqual((folder/'gate.json').read_bytes(),(out/'gate.json').read_bytes())
            proof=json.loads((folder/'recovered_gate_provenance.json').read_text())
            self.assertEqual(proof['attempt_gate'],str(out/'gate.json'))
            self.assertEqual((folder/'original_failure.log').read_text(),'original failure')
            publish_recovered_gate(entry,root)
            self.assertFalse(apply_retry_request(entry,request,root,{}))

    def test_active_or_nonlauncher_failures_are_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,entry,request=self.fixture(root)
            self.assertFalse(apply_retry_request(entry,request,root,{'456':'RUNNING'}))
            entry['status']='replay_failed'
            self.assertFalse(apply_retry_request(entry,request,root,{}))

    def test_existing_gate_or_changed_job_identity_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder,entry,request=self.fixture(root)
            bad={**request,'expected_replay_job_id':'other'}
            with self.assertRaisesRegex(ValueError,'identity differs'):apply_retry_request(entry,bad,root,{})
            (folder/'gate.json').write_text('{"success":false}')
            with self.assertRaisesRegex(ValueError,'existing canonical'):apply_retry_request(entry,request,root,{})
            self.assertEqual((folder/'gate.json').read_text(),'{"success":false}')

    def test_changed_canonical_gate_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder,entry,request=self.fixture(root)
            apply_retry_request(entry,request,root,{})
            out=Path(entry.pop('retry_output'));out.mkdir();(out/'gate.json').write_text('{"success":true}')
            entry.update(output=str(out),replay_job_id='789',gate={'success':True})
            (folder/'gate.json').write_text('immutable different evidence')
            with self.assertRaisesRegex(ValueError,'different evidence'):publish_recovered_gate(entry,root)
            self.assertEqual((folder/'gate.json').read_text(),'immutable different evidence')

if __name__=='__main__':unittest.main()
