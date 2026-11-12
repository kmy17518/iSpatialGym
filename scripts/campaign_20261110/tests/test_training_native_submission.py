import hashlib
import json
import multiprocessing
from pathlib import Path
import tempfile
import unittest

from scripts.campaign_20261110 import training_native_submission as S


def contender(args):
    root, request, digest, marker = args
    try:
        claim = S.acquire(root, request, digest, 0)
    except S.AlreadyClaimed:
        return False
    # Represents parser/allocation work: only the claim winner may enter.
    with Path(marker).open('a') as stream:
        stream.write('entered\n')
    claim.transition('preflight_passed', preflight={'passed': True})
    claim.transition('submitting', command=['sbatch', '--parsable', 'exact.sh'])
    return True


class NativeSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'campaign_20261110'
        self.root.mkdir()
        row = {'candidate_id': 'a', 'start': 50296900, 'count': 1,
               'native_runtime': str(self.root/'code/native'), 'output': str(self.root/'native')}
        for field, hash_field in [('config_path','config_sha256'), ('authorization_path','authorization_sha256'),
                                  ('input_manifest','input_manifest_sha256')]:
            p = self.root / (field+'.json'); p.write_text('{}\n')
            row[field] = str(p); row[hash_field] = S.digest(p)
        self.request = self.root/'request.json'
        self.request.write_text(json.dumps({'campaign_id': self.root.name, 'records': [row, {**row, 'candidate_id':'b', 'start':50297000}]}))
        self.sha = S.digest(self.request)

    def claim(self, member=0):
        return S.acquire(self.root, self.request, self.sha, member)

    def test_concurrent_callbacks_enter_parser_once_and_crash_does_not_release(self):
        marker = self.root/'parser.log'
        with multiprocessing.get_context('fork').Pool(8) as pool:
            results = pool.map(contender, [(str(self.root),str(self.request),self.sha,str(marker))]*16)
        self.assertEqual(sum(results), 1)
        self.assertEqual(marker.read_text(), 'entered\n')
        with self.assertRaises(S.AlreadyClaimed): self.claim()
        state = json.loads(next((self.root/'infra/training_native_submissions').glob('*/state.json')).read_text())
        self.assertEqual(state['phase'], 'submitting')
        self.assertFalse(state['automatic_release_allowed'])

    def test_submitted_callback_never_resubmits(self):
        c = self.claim(); c.transition('preflight_passed'); c.transition('submitting', command=['sbatch','exact'])
        c.transition('submitted', returncode=0, job_id='18456100')
        with self.assertRaises(S.AlreadyClaimed): self.claim()

    def test_member_claims_independent(self):
        self.assertNotEqual(self.claim(0).directory, self.claim(1).directory)

    def test_changed_request_and_payload_rejected_before_claim(self):
        with self.assertRaisesRegex(ValueError, 'request bytes'): S.acquire(self.root,self.request,'0'*64,0)
        (self.root/'config_path.json').write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, 'identity changed'): self.claim()
        self.assertFalse((self.root/'infra/training_native_submissions').exists())

    def test_orphan_claim_directory_is_not_taken_over(self):
        c = self.claim(); (c.directory/'state.json').unlink()
        with self.assertRaises(S.AlreadyClaimed): self.claim()

    def test_failed_preflight_retains_claim(self):
        c = self.claim(); c.transition('blocked', error='parser rejected')
        with self.assertRaises(S.AlreadyClaimed): self.claim()

    def test_allocating_requires_preflight_and_command(self):
        c = self.claim()
        with self.assertRaises(ValueError): c.transition('submitting', command=['sbatch'])
        c.transition('preflight_passed')
        with self.assertRaises(ValueError): c.transition('submitting')
        c.transition('submitting', command=['sbatch','exact'])
        with self.assertRaises(ValueError): c.transition('submitted', returncode=1,job_id='')
        c.transition('submission_unresolved', returncode=1, stderr='connection lost')
        with self.assertRaises(S.AlreadyClaimed): self.claim()

    def test_owner_token_cannot_change(self):
        c = self.claim(); c.token = 'another-owner'
        with self.assertRaisesRegex(ValueError, 'owner changed'): c.transition('preflight_passed')


if __name__ == '__main__': unittest.main()
