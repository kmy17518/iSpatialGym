import json
from pathlib import Path
import tempfile
import unittest
from scripts.campaign_20261110.infra_gpu_accounting import gpu_count,native_evidence


class AccountingTests(unittest.TestCase):
    def test_gpu_total_does_not_double_count_typed_tres(self):
        self.assertEqual(gpu_count('cpu=8,gres/gpu=2,gres/gpu:rtx4090=2'),2)
        self.assertEqual(gpu_count('cpu=8,gres/gpu:rtx4090=1,gres/gpu:rtx5090=1'),2)
        self.assertEqual(gpu_count('cpu=1,mem=2G'),0)

    def test_native_failures_remain_in_denominator_and_missing_stats_stay_unknown(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            self.assertIsNone(native_evidence(root)['job_stats'])
            (root/'job_stats.json').write_text(json.dumps({'attempts':9,'written':2,'world_load_s':80}))
            for name,success in [('one',True),('two',False)]:
                p=root/'demo'/name/'meta.json';p.parent.mkdir(parents=True)
                p.write_text(json.dumps({'success':success,'status':'success' if success else 'failure',
                    'started_utc':'2026-11-10T15:00:00Z','wall_time_s':12}))
            data=native_evidence(root)
            self.assertEqual(data['job_stats']['attempts'],9)
            self.assertEqual(data['demo_reports'],2);self.assertEqual(data['demo_successes'],1)
            self.assertEqual(data['last_demo_completed_unix']-data['first_demo_started_unix'],12)


if __name__=='__main__':unittest.main()
