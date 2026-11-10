import json
import concurrent.futures
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
        self.assertEqual(D.option(cmd, '--ledger'), str(self.root / 'manifests/dispatch_ledgers/fresh/allocations.tsv'))
        self.assertEqual(env['ISG_DISPATCH_CAMPAIGN_ROOT'], str(self.root))

    def test_private_ledger_keeps_lower_reserved_range_after_later_intent(self):
        first = D.prepare_intent_ledger(self.root, 'first')
        second = D.prepare_intent_ledger(self.root, 'second')
        with second.open('a') as stream:
            stream.write('now\tobject_scaling-apple\talias\t50009900\t3\n')
        self.assertNotEqual(first, second)
        with first.open() as stream:
            self.assertEqual(list(D.csv.DictReader(stream, delimiter='\t')), [])
        self.assertEqual(D.prepare_intent_ledger(self.root, 'first'), first)
        self.assertEqual((self.root/'manifests/dispatch_first_allocations.tsv').resolve(), first.resolve())
        self.assertEqual(first.stat().st_mode & 0o777, 0o660)

    def test_shifted_dry_run_cannot_reach_scheduler(self):
        identity = {'start':50000000, 'n':3}
        D.verify_dry_indices('  START=50000000\n  N=3\n', identity)
        for text in ['START=50009903\nN=3\n', 'START=50000000\nN=4\n',
                     'START=50000000\nSTART=50000000\nN=3\n']:
            with self.assertRaisesRegex(ValueError, 'shifted'):
                D.verify_dry_indices(text, identity)

    def test_concurrent_ledger_preparation_publishes_one_complete_header(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            paths = list(pool.map(lambda _: D.prepare_intent_ledger(self.root, 'same'), range(32)))
        self.assertEqual(len(set(paths)), 1)
        self.assertEqual(paths[0].read_text(), '\t'.join(D.LEDGER_COLS) + '\n')
        self.assertEqual((self.root/'manifests/dispatch_same_allocations.tsv').resolve(), paths[0].resolve())

    def test_capacity_counts_gpu_arrays_and_cpu_job_caps(self):
        rows = [['shiyuc', 'svl', 'RUNNING', 'gres/gpu:titanrtx:1']] * 46
        rows += [['shiyuc', 'svl', 'RUNNING', 'N/A']] * 8
        self.assertEqual(D.backend_capacity('shiyuc_svl', rows), 2)
        self.assertEqual(D.backend_capacity('minyeong_svl', rows), 8)
        self.assertEqual(D.backend_capacity('vast_5090', rows), 8)

    def test_shared_ledger_symlink_reconciles_exact_new_allocation(self):
        target = self.root / 'manifests/shiyuc/allocations.tsv'; target.parent.mkdir()
        target.write_text('key\tstart\tn\tcode\ttag\tjobid\tjob_dir\nobject_scaling-apple\t50000000\t3\t' + str(self.code) + '\theld_queue\t123\t' + str(self.root / 'out/shiyuc/out/held_queue/job') + '\n')
        (self.root / 'manifests/shiyuc_allocations.tsv').symlink_to(target)
        row = D.find_allocation(self.root, D.input_identity(self.root, self.command))
        self.assertEqual(row['jobid'], '123')

    def test_empty_index_is_validated_hashed_and_group_readable(self):
        path = self.root / 'empty_eval_index.json'
        path.write_text('{"schema":"isg-instance-index-v1","n":0,"instances":[]}')
        path.chmod(0o600)
        command = self.command + ['--extra-env', 'EVAL_INDEX=' + str(path)]
        payload = json.loads(D.enqueue(self.root, 'index', command).read_text())
        self.assertIn(str(path), payload['identity']['files'])
        self.assertTrue(path.stat().st_mode & 0o040)
        path.write_text('{"schema":"wrong"}')
        with self.assertRaises(ValueError): D.enqueue(self.root, 'bad_index', command)


if __name__ == '__main__': unittest.main()
