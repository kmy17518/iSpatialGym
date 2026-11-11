import json
from pathlib import Path
import tempfile
import unittest

from scripts.campaign_20261110.ik_observation import audit, CAMPAIGN


class IKObservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / CAMPAIGN
        self.job = self.root / 'out/source/class/g1'
        self.job.mkdir(parents=True)
        self.manifest = self.root / 'manifest.json'
        self.manifest.write_text(json.dumps({'compatibility_sha256': 'a' * 64}))
        self.metrics = self.root / 'metrics'
        self.metrics.mkdir()
        (self.job / 'job.env').write_text(
            f'ISPATIALGYM_EXACT_IK_MANIFEST={self.manifest}\n'
            f'ISPATIALGYM_EXACT_IK_LOCAL_CACHE={self.root}/cache\n'
            f'ISPATIALGYM_EXACT_IK_METRICS={self.metrics}\n'
            'ISPATIALGYM_EXACT_IK_PRODUCER=minyeong-test\nAPI_KEY=PRIVATE\n')
        self.reg = self.root / 'ik_exchange/registered_jobs/1.json'
        self.reg.parent.mkdir(parents=True)
        self.reg.write_text(json.dumps({'campaign_id': CAMPAIGN, 'account': 'minyeong',
                                       'job_id': '1', 'job_dir': str(self.job)}))

    def tearDown(self):
        self.temp.cleanup()

    def test_enabled_without_runtime_never_invents_hits_or_other_account_zeros(self):
        value = audit(self.root)
        self.assertTrue(value['jobs'][0]['enabled'])
        self.assertFalse(value['jobs'][0]['runtime_compatibility_verified'])
        self.assertIsNone(value['jobs'][0]['counters'])
        self.assertTrue(all(a['counters'] is None for a in value['accounts']))
        self.assertNotIn('PRIVATE', json.dumps(value))

    def test_measured_hits_and_actual_runtime_cache_preserve_denominator(self):
        (self.metrics / '31.json').write_text(json.dumps({'campaign_id': CAMPAIGN,
            'producer': 'minyeong-test', 'pid': 31, 'compatibility_sha256': 'a' * 64,
            'local_cache': str(self.root / 'cache/nested/actual'), 'observed_at': '2026-11-11T11:00:00Z',
            'counters': {'exact_queries': 10, 'exact_cache_eligible_queries': 8,
                         'exact_shared_verified_hit': 3, 'exact_fresh_solver_calls': 19}}))
        value = audit(self.root); row = value['jobs'][0]
        self.assertTrue(row['runtime_compatibility_verified'])
        self.assertEqual(row['counters']['exact_shared_verified_hit'], 3)
        self.assertEqual(row['counters']['exact_cache_eligible_queries'], 8)
        self.assertEqual(row['actual_local_caches'], [str(self.root / 'cache/nested/actual')])
        self.assertIsNone(row['sampling_time_benefit_seconds'])

    def test_foreign_metric_is_rejected_and_not_counted(self):
        (self.metrics / '31.json').write_text(json.dumps({'campaign_id': 'previous', 'counters': {'exact_verified_hit': 999}}))
        row = audit(self.root)['jobs'][0]
        self.assertIsNone(row['counters']); self.assertTrue(row['errors'])

    def test_external_registered_path_is_rejected(self):
        value = json.loads(self.reg.read_text()); value['job_dir'] = str(Path(self.temp.name) / 'other')
        self.reg.write_text(json.dumps(value))
        self.assertEqual(audit(self.root)['jobs'], [])
        self.assertTrue(audit(self.root)['errors'])


if __name__ == '__main__':
    unittest.main()
