import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.campaign_20261110 import audit_selected_libraries as A


class ObservationFailureTests(unittest.TestCase):
    def test_unreadable_output_does_not_abort_or_become_physical_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = []
            for name in ('unreadable', 'empty'):
                job = dict(id=name, category='test_category', model=name, dataset='test_dataset',
                           scale=[1., 1., 1.], constraint='tabletop', ready=True,
                           readiness_source='test source', asset_sha256='a' * 64)
                jobs.append(job)
                path = root / 'models' / name / 'library.json'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({**job, 'schema': 'isg-grasp-library/2',
                    'provenance': {'validator': 'isg-tabletop-validator/1'},
                    'physics_profile': {'test': True}, 'policy_key': 'test', 'grasps': []}))
            manifest = root / 'jobs.json'
            manifest.write_text(json.dumps(jobs))
            inputs = [{'manifest': A.reference(manifest), 'indices': [0, 1],
                       'library_root': str(root)}]
            original_read = Path.read_text

            def read(path, *args, **kwargs):
                if path == root / 'models/unreadable/library.json':
                    raise PermissionError('secondary atomic output is private')
                return original_read(path, *args, **kwargs)

            policy = ({}, {'train_exclude_models': [], 'category_holdout_categories': []}, {})
            with patch.object(A, 'active_policy', return_value=policy), patch.object(Path, 'read_text', read):
                result = A.audit(root, inputs)
            self.assertEqual(result['summary'], dict(attempted_entries=2, verified=0, failed=1, unavailable=1))
            self.assertEqual(result['unavailable'][0]['entry_id'], 'unreadable')
            self.assertIsNone(result['unavailable'][0]['library']['sha256'])
            self.assertEqual(result['failed'][0]['entry_id'], 'empty')
            self.assertEqual(result['failed'][0]['reason'], 'zero positive tested grasps')


if __name__ == '__main__':
    unittest.main()
