import hashlib
from pathlib import Path
import tempfile
import unittest

from scripts.campaign_20261110.training_policy_history import preserve, replace_policy


class PolicyHistoryTests(unittest.TestCase):
    def test_original_format_and_both_identities_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / 'controllers/training_production_gate.json'
            policy.parent.mkdir()
            original = b'{ "allowed_canaries": [], "standalone_training_allowed": false }\n'
            policy.write_bytes(original)
            result = replace_policy(root, {'allowed_canaries': [], 'standalone_training_allowed': False,
                                          'observed_at': 'later'},
                                    expected_previous_sha256=hashlib.sha256(original).hexdigest())
            self.assertEqual(Path(result['previous']['path']).read_bytes(), original)
            self.assertEqual(Path(result['current']['path']).read_bytes(), policy.read_bytes())
            self.assertEqual(preserve(root, original), result['previous'])

    def test_changed_policy_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / 'controllers/training_production_gate.json'
            policy.parent.mkdir()
            policy.write_bytes(b'current')
            with self.assertRaisesRegex(ValueError, 'policy changed'):
                replace_policy(root, {}, expected_previous_sha256='0' * 64)
            self.assertEqual(policy.read_bytes(), b'current')

    def test_corrupted_content_address_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ref = preserve(root, b'original')
            Path(ref['path']).write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'snapshot differs'):
                preserve(root, b'original')


if __name__ == '__main__':
    unittest.main()
