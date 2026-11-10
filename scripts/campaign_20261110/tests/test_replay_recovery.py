import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from scripts.campaign_20261110.controller_validation import apply_retry_request, publish_recovered_gate, ledger_paths

class ReplayRecoveryTests(unittest.TestCase):
    def test_dispatch_ledgers_are_discovered_once_with_explicit_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target=root/'manifests/dispatch_ledgers/token/allocations.tsv'
            target.parent.mkdir(parents=True);target.write_text('header\n')
            link=root/'manifests/dispatch_token_allocations.tsv';link.symlink_to(target)
            duplicate=root/'manifests/dispatch_other_allocations.tsv';duplicate.symlink_to(target)
            paths=ledger_paths(root,[target])
            self.assertEqual([p.resolve() for p in paths],[target])
            self.assertIn(target,[p.resolve() for p in ledger_paths(root)])

    def test_dispatch_ledger_must_stay_in_current_campaign(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'manifests').mkdir()
            (root/'manifests/dispatch_bad_allocations.tsv').symlink_to('/outside/allocations.tsv')
            with self.assertRaisesRegex(ValueError,'outside owned'):ledger_paths(root)

    def metadata_fixture(self, root):
        folder,entry,request=self.fixture(root)
        job=root/'out/fresh';job.mkdir(parents=True)
        (job/'sampling.done').touch()
        metadata=job/'template_meta.json';metadata.write_text('{"closed":true}')
        audit=root/'preflight.json';audit.write_text('{"strict_preflight":"passed"}')
        log=folder/'scene_verify.log';log.write_text('ValueError: source template metadata changed')
        gate={'success':False,'replay_rc':5,'replay_mode':'physics_action_open_loop',
              'source_job':str(job),'result':{'success':True,'qa_hard':[],'goal_override':None}}
        path=folder/'gate.json';path.write_text(json.dumps(gate))
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        entry.update(status='replay_failed',source_job_dir=str(job),gate=gate)
        request.update(failure_kind='mutable_source_metadata',expected_failed_gate_sha256=sha(path),
            scene_verify_log_sha256=sha(log),source_metadata=str(metadata),source_metadata_sha256=sha(metadata),
            preflight_audit=str(audit),preflight_audit_sha256=sha(audit))
        return folder,entry,request

    def test_metadata_race_preserves_failed_bytes_and_still_requires_new_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder,entry,request=self.metadata_fixture(root)
            old=(folder/'gate.json').read_bytes()
            self.assertTrue(apply_retry_request(entry,request,root,{}))
            self.assertEqual((folder/'failed_gate_before_02.json').read_bytes(),old)
            self.assertFalse((folder/'gate.json').exists())
            out=Path(entry.pop('retry_output'));out.mkdir()
            entry.update(output=str(out),replay_job_id='789',gate={'success':False})
            (out/'gate.json').write_text('{"success":false}')
            publish_recovered_gate(entry,root)
            self.assertFalse((folder/'gate.json').exists())
            entry['gate']={'success':True};(out/'gate.json').write_text('{"success":true}')
            publish_recovered_gate(entry,root)
            self.assertEqual((folder/'failed_gate_before_02.json').read_bytes(),old)
            self.assertEqual(json.loads((folder/'gate.json').read_text()),{'success':True})

    def test_metadata_retry_rejects_changed_hash_live_source_or_physics_failure(self):
        for failure in ('hash','live','physics'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);folder,entry,request=self.metadata_fixture(root)
                active={}
                if failure=='hash':request['expected_failed_gate_sha256']='0'*64
                if failure=='live':active={'123':'RUNNING'}
                if failure=='physics':
                    gate=json.loads((folder/'gate.json').read_text());gate['result']['success']=False
                    (folder/'gate.json').write_text(json.dumps(gate))
                    request['expected_failed_gate_sha256']=hashlib.sha256((folder/'gate.json').read_bytes()).hexdigest()
                with self.assertRaises(ValueError):apply_retry_request(entry,request,root,active)
                self.assertTrue((folder/'gate.json').exists())
                self.assertFalse((folder/'failed_gate_before_02.json').exists())

    def test_metadata_archive_is_reconciled_after_interruption_before_state_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder,entry,request=self.metadata_fixture(root)
            old_entry=dict(entry)
            apply_retry_request(entry,request,root,{})
            self.assertTrue(apply_retry_request(old_entry,request,root,{}))
            self.assertEqual(len(old_entry['attempt_history']),1)

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
