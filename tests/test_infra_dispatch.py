import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.campaign_20261110 import infra_dispatch as D


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.code = self.root / 'code/trunk_test'
        self.code.mkdir(parents=True)
        (self.code / '.snapshot_commit').write_text('immutable-code')
        (self.root / 'manifests').mkdir()
        (self.root / 'manifests/family_index_reservations.json').write_text(json.dumps({'reservations': {'fresh': {'start': 50000000, 'n': 3}}}))
        self.config = self.root / 'config.json'; self.config.write_text('{}')
        self.command = ['python3', str(self.code / 'scripts/trainfast/submit_fast.py'), 'submit', 'object_scaling-apple', '3',
                        '--root', str(self.root), '--code', str(self.code), '--base-index', '50000000', '--tag', 'held_queue',
                        '--extra-env', 'ISPATIALGYM_CAMPAIGN_CONFIG=' + str(self.config), '--extra-env', 'ISGF_COLLECT_EXTRA=--no-goal-override']

    def test_immutable_identity_and_idempotent_enqueue(self):
        path = D.enqueue(self.root, 'fresh', self.command, backends=['vast_5090'])
        self.assertEqual(path, D.enqueue(self.root, 'fresh', self.command, backends=['vast_5090']))
        payload = json.loads(path.read_text())
        self.config.write_text('{"changed": true}')
        with self.assertRaises(ValueError): D.routed_command(self.root, payload, 'vast_5090')

    def test_indices_must_be_reserved_and_goals_preserved(self):
        cmd = self.command.copy(); D.set_option(cmd, '--base-index', 50001000)
        with self.assertRaises(ValueError): D.enqueue(self.root, 'outside', cmd)
        cmd = self.command.copy(); cmd[-1] = 'ISGF_COLLECT_EXTRA='
        with self.assertRaises(ValueError): D.enqueue(self.root, 'override', cmd)

    def test_ambiguous_submission_is_not_retried(self):
        path = D.enqueue(self.root, 'fresh', self.command, backends=['vast_5090'])
        D.shared_json(self.root / 'dispatch/receipts/fresh.json', {'status': 'submitting'})
        with patch.object(D.subprocess, 'run') as run:
            with self.assertRaises(ValueError): D.execute(self.root, path, 'vast_5090')
            run.assert_not_called()

    def test_shiyuc_keeps_canonical_root_and_distinct_write_paths(self):
        (self.root / 'out/shiyuc').mkdir(parents=True)
        payload = json.loads(D.enqueue(self.root, 'fresh', self.command).read_text())
        with patch.object(D.getpass, 'getuser', return_value='shiyuc'):
            cmd, env = D.routed_command(self.root, payload, 'shiyuc_viscam')
        self.assertEqual(D.option(cmd, '--root'), str(self.root / 'out/shiyuc'))
        self.assertIn('ISGF_ROOT=' + str(self.root), cmd)
        self.assertIn('ISGF_SAMPLING_CACHE=/vision/u/shiyuc/minyeong/campaign_20261110/cache/dispatch/fresh', cmd)
        self.assertEqual(D.option(cmd, '--ledger'), str(self.root / 'manifests/shiyuc/allocations.tsv'))
        self.assertEqual(env['ISG_DISPATCH_CAMPAIGN_ROOT'], str(self.root))

    def test_capacity_counts_gpu_arrays_and_cpu_job_caps(self):
        rows = [['shiyuc', 'svl', 'RUNNING', 'gres/gpu:titanrtx:1']] * 46
        rows += [['shiyuc', 'svl', 'RUNNING', 'N/A']] * 8
        self.assertEqual(D.backend_capacity('shiyuc_svl', rows), 2)
        self.assertEqual(D.backend_capacity('minyeong_svl', rows), 8)
        self.assertEqual(D.backend_capacity('vast_5090', rows), 8)


if __name__ == '__main__': unittest.main()
