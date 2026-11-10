import json
from pathlib import Path
import tempfile
import unittest
from scripts.campaign_20261110.infra_vast.lifecycle import retain_attempt
from scripts.campaign_20261110.infra_gpu_accounting import attempt_accounting


class AttemptLifecycleTests(unittest.TestCase):
    def test_zero_attempt_failure_survives_retry_and_missing_intervals_are_not_estimated(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);directory=root/'vast/state/attempts'
            first={'id':'9000001','instance':'one','assigned':10,'tries':1}
            done={'attempt_id':'first','attempts':0,'rc':1,'gpu_count':1,'fetch_started_unix':11,'upload_finished_unix':21}
            evidence=retain_attempt(directory,first,done)
            self.assertEqual(evidence,retain_attempt(directory,first,done))
            with self.assertRaisesRegex(ValueError,'changed'):retain_attempt(directory,first,{**done,'rc':0})
            state={'status':'synced','tries':2}
            self.assertIsNone(attempt_accounting(root,'9000001',state)['complete_worker_gpu_seconds_all_attempts'])
            retain_attempt(directory,{**first,'instance':'two','assigned':22,'tries':2},
                {'attempt_id':'second','attempts':5,'rc':0,'gpu_count':1,'fetch_started_unix':23,'upload_finished_unix':43})
            result=attempt_accounting(root,'9000001',state)
            self.assertEqual(result['complete_worker_gpu_seconds_all_attempts'],30)
            self.assertEqual(result['retained_native_attempts_all_tries'],5)
            self.assertEqual(json.loads(Path(evidence['path']).read_text())['done']['rc'],1)


if __name__=='__main__':unittest.main()
