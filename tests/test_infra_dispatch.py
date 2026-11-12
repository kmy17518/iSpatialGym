import json
import concurrent.futures
from pathlib import Path
import tempfile
import unittest
from contextlib import contextmanager
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
        self.config = self.root / 'config.json'; self.config.write_text(json.dumps({'template':{'split':'eval_standard','task':'object_scaling','class_id':'apple'}}))
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

    def test_nested_public_snapshot_dataset_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            provider=Path(directory);provider.chmod(0o755)
            data=provider/'datasets';data.mkdir();leaf=data/'scene.json';leaf.write_text('{"scene":"exact"}');leaf.chmod(0o644)
            (self.code/'Behavior-1K').symlink_to(provider,target_is_directory=True)
            logical=self.code/'Behavior-1K/datasets/scene.json'
            value=json.loads(self.config.read_text());value['source_scene']=str(logical);self.config.write_text(json.dumps(value))
            before=leaf.stat().st_ctime_ns
            D.enqueue(self.root,'public_dataset',self.command,backends=['shiyuc_svl'])
            self.assertEqual(leaf.stat().st_ctime_ns,before)
            self.assertEqual(leaf.read_text(),'{"scene":"exact"}')
            leaf.chmod(0o600)
            with self.assertRaisesRegex(ValueError,'shared input outside'):
                D.enqueue(self.root,'private_dataset',self.command,backends=['shiyuc_svl'])
            leaf.chmod(0o644);provider.chmod(0o700)
            with self.assertRaisesRegex(ValueError,'shared input outside'):
                D.enqueue(self.root,'unsearchable_dataset',self.command,backends=['shiyuc_svl'])

    def test_arbitrary_external_symlink_is_not_a_shared_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            leaf=Path(directory)/'private.json';leaf.write_text('{}');leaf.chmod(0o644)
            link=self.root/'external.json';link.symlink_to(leaf)
            value=json.loads(self.config.read_text());value['source_scene']=str(link);self.config.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'shared input outside'):
                D.enqueue(self.root,'external',self.command,backends=['shiyuc_svl'])

    def test_snapshot_can_mount_one_dataset_without_mounting_its_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            provider=Path(directory);provider.chmod(0o755)
            data=provider/'task_instances';data.mkdir();leaf=data/'scene.json';leaf.write_text('{}');leaf.chmod(0o644)
            datasets=self.code/'Behavior-1K/datasets';datasets.mkdir(parents=True)
            (datasets/'task_instances').symlink_to(data,target_is_directory=True)
            logical=datasets/'task_instances/scene.json'
            self.assertTrue(D.public_snapshot_dataset_leaf(self.root,logical,logical.resolve()))
            other=provider/'outside.json';other.write_text('{}');other.chmod(0o644)
            (data/'escape.json').symlink_to(other)
            escape=datasets/'task_instances/escape.json'
            self.assertFalse(D.public_snapshot_dataset_leaf(self.root,escape,escape.resolve()))

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

    def test_private_dry_run_lock_repaired_without_replacing_inode(self):
        ledger = D.prepare_intent_ledger(self.root, 'dry_lock')
        lock = ledger.with_name(ledger.name + '.lock')
        lock.write_text('existing lock identity')
        lock.chmod(0o600)
        inode = lock.stat().st_ino
        D.prepare_intent_ledger(self.root, 'dry_lock')
        self.assertEqual(lock.stat().st_ino, inode)
        self.assertEqual(lock.read_text(), 'existing lock identity')
        self.assertEqual(lock.stat().st_mode & 0o660, 0o660)

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

    def test_capacity_uses_partition_qos_when_job_qos_is_normal(self):
        rows = [['minyeong','svl','RUNNING','N/A','normal']] * 56
        rows += [['minyeong','viscam','RUNNING','N/A','normal']] * 60
        rows += [['minyeong','svl,viscam','PENDING','gres/gpu:1','normal']] * 23
        self.assertEqual(D.backend_capacity('minyeong_svl', rows), 0)
        self.assertEqual(D.backend_capacity('minyeong_viscam', rows), 7)
        self.assertEqual(D.backend_capacity('shiyuc_viscam', rows), 8)

    def test_submission_retry_preserves_prior_readonly_logs(self):
        receipt = self.root / 'receipt.json'; prior = self.root/'receipt.dry.log'
        prior.write_text('prior account evidence');prior.chmod(0o440)
        log = D.submission_log(receipt,'.dry.log','fresh dry-run')
        self.assertNotEqual(log,prior)
        self.assertEqual(prior.read_text(),'prior account evidence')
        self.assertEqual(log.read_text(),'fresh dry-run')
        self.assertEqual(log.stat().st_mode & 0o777,0o660)

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

    def test_enqueue_preserves_already_readable_immutable_metadata(self):
        paths = [self.code/'.snapshot_commit', self.config,
                 self.root/'manifests/family_index_reservations.json']
        for path in paths:
            path.chmod(0o640)
        before = {path: (path.stat().st_ctime_ns, D.file_digest(path)) for path in paths}
        real_chmod = D.os.chmod
        with patch.object(D.os, 'chmod', wraps=real_chmod) as chmod:
            D.enqueue(self.root, 'readable', self.command, backends=['shiyuc_svl'])
        changed = {Path(call.args[0]) for call in chmod.call_args_list}
        self.assertFalse(set(paths) & changed)
        self.assertEqual(before, {path: (path.stat().st_ctime_ns, D.file_digest(path)) for path in paths})

    def test_vast_activation_retries_contention_without_resubmission(self):
        path = D.enqueue(self.root, 'fresh', self.command, backends=['vast_5090'])
        payload = json.loads(path.read_text())
        D.shared_json(self.root/'vast/config.json', {})
        original = D.atomic_directory_lock
        attempts = []
        @contextmanager
        def busy_once(path):
            attempts.append(path)
            if len(attempts) == 1: raise FileExistsError(path)
            with original(path): yield
        with patch.object(D, 'atomic_directory_lock', busy_once), patch.object(D.time, 'sleep'):
            D.activate_vast(self.root, payload, {'jobid':'900001'})
        self.assertEqual(len(attempts), 2)
        self.assertEqual(json.loads((self.root/'vast/config.json').read_text())['dispatch_job_ids'], ['900001'])

    def test_allocated_blocked_receipt_only_reconciles_activation(self):
        path = D.enqueue(self.root, 'fresh', self.command, backends=['vast_5090'])
        row = {'jobid':'900001'}
        D.shared_json(self.root/'dispatch/receipts/fresh.json',
                      {'status':'blocked','backend':'vast_5090','allocation':row,'error':'lock contention'})
        with patch.object(D, 'find_allocation', return_value=row), patch.object(D, 'activate_vast') as activate, patch.object(D.subprocess, 'run') as run:
            D.execute(self.root, path, 'vast_5090')
        run.assert_not_called(); activate.assert_called_once()
        receipt = json.loads((self.root/'dispatch/receipts/fresh.json').read_text())
        self.assertEqual(receipt['status'], 'submitted')
        self.assertEqual(receipt['reconciliation_history'][0]['previous_error'], 'lock contention')


if __name__ == '__main__': unittest.main()
