from pathlib import Path
import json
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from scripts.campaign_20261110 import controller_families as c


class FamilyControllerTests(unittest.TestCase):
    def recovered_fixture(self, root, *, group=False, mode='physics_action_open_loop'):
        job = root / 'out/fresh'
        job.mkdir(parents=True)
        row = {'key': 'a', 'jobid': '12', 'job_dir': str(job)}
        stage = {'status': 'failed', 'allocation': row, 'directory': str(root/'groups/g'), 'index': 50000000}
        state = {'history': [{'name': 'g', 'group' if group else 'anchor': stage}], 'consecutive_failures': 3}
        path = c.gate_path(root, row)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'success': True, 'replay_mode': mode, 'source_job': str(job)}))
        return state, path

    def test_late_physics_recovery_resumes_without_overwriting_failed_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state, path = self.recovered_fixture(root)
            self.assertTrue(c.restore_recovered_cycle({'class_id': 'a'}, state, root))
            self.assertEqual(state['current']['anchor']['status'], 'waiting')
            self.assertEqual(state['history'][0]['anchor']['status'], 'failed')
            self.assertEqual(state['current']['recovery_evidence']['gate_sha256'], c.file_digest(path))
            self.assertEqual(state['consecutive_failures'], 0)
            state.pop('current')
            self.assertFalse(c.restore_recovered_cycle({'class_id': 'a'}, state, root))

    def test_recovery_preserves_group_identity_for_strict_reaudit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state, _ = self.recovered_fixture(root, group=True)
            self.assertTrue(c.restore_recovered_cycle({'class_id': 'a'}, state, root))
            self.assertEqual(state['current']['name'], 'g')
            self.assertEqual(state['current']['group']['allocation']['jobid'], '12')

    def test_recovery_never_uses_kinematic_gates_or_overrides_disabled_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state, path = self.recovered_fixture(root, mode='kinematic')
            self.assertFalse(c.restore_recovered_cycle({'class_id': 'a'}, state, root))
            data = json.loads(path.read_text()); data['replay_mode'] = 'physics_action_open_loop'; path.write_text(json.dumps(data))
            self.assertFalse(c.restore_recovered_cycle({'class_id': 'a', 'allow_new_cycles': False}, state, root))
            data['source_job'] = '/outside'; path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                c.restore_recovered_cycle({'class_id': 'a'}, state, root)

    def test_spooled_group_does_not_submit_directly(self):
        record = {'status': 'ready', 'directory': '/campaign/groups/unique', 'index': 50000000}
        with patch('scripts.campaign_20261110.controller_spool.enqueue_record') as enqueue, patch.object(c, 'run_logged') as submit:
            c.perform_submit(record, ['python', 'submit_fast.py', 'submit'], lambda: None, [], {'class_id': 'a'},
                Path('/campaign'), 'goal_groups', '/snapshot', 3, dispatch_spool=True)
            enqueue.assert_called_once()
            submit.assert_not_called()
            self.assertEqual(record['status'], 'external_submission')
            self.assertTrue(record['dry_run_passed'])

    def test_bucket_deficits_are_strict_and_stop_at_quota(self):
        self.assertEqual(c.next_bucket({("eval_standard", 0): 3}, [3, 4, 3]), 1)
        self.assertIsNone(c.next_bucket({("eval_standard", 0): 3, ("eval_standard", 1): 4,
                                        ("eval_standard", 2): 3, ("train", 0): 999}, [3, 4, 3]))

    def test_global_indices_skip_other_ledger_and_are_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifests").mkdir()
            (root / "manifests/other_allocations.tsv").write_text("start\tn\n50000000\t3\n")
            first = c.reserve_indices(root, "one", 3)
            self.assertEqual(first, 50000100)
            self.assertEqual(c.reserve_indices(root, "one", 3), first)
            self.assertGreaterEqual(c.reserve_indices(root, "two", 1), first + 3)
            with self.assertRaises(ValueError):
                c.reserve_indices(root, "one", 2)

    def test_ambiguous_submit_never_retries(self):
        record = {"status": "submitting", "index": 50000000}
        with patch.object(c, "run_logged") as submit:
            with self.assertRaises(ValueError):
                c.perform_submit(record, ["sbatch"], lambda: None, [], {"class_id": "class"},
                                 Path("/tmp"), "goal_groups", "/snapshot", 3)
            submit.assert_not_called()

    def test_reconcile_after_crash_adopts_exact_ledger_allocation(self):
        row = {"key": "class", "start": "50000000", "n": "3", "code": "/snapshot", "tag": "goal_groups", "jobid": "12"}
        record = {"status": "submitting", "index": 50000000}
        with patch.object(c, "run_logged") as submit:
            c.perform_submit(record, ["sbatch"], lambda: None, [row], {"class_id": "class"},
                             Path("/tmp"), "goal_groups", "/snapshot", 3)
            submit.assert_not_called()
        self.assertEqual(record["allocation"]["jobid"], "12")
        self.assertEqual(record["status"], "waiting")
        with self.assertRaises(ValueError):
            c.reconcile([row], key="class", index=50000000, count=2, snapshot="/snapshot", tag="goal_groups")

    def test_capacity_counts_unrelated_jobs_and_reserves_validation_slots(self):
        queue = {str(i): [str(i), "RUNNING", "minyeong", "svl", "gres/gpu:titanrtx:1"] for i in range(43)}
        self.assertEqual(c.submission_capacity(queue, 8), 1)
        queue["43"] = ["43", "RUNNING", "minyeong", "svl", "gres/gpu:titanrtx:1"]
        self.assertEqual(c.submission_capacity(queue, 8), 0)

    def test_counts_only_committed_matching_class_database_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "dataset/groups").mkdir(parents=True)
            (root / "dataset/groups/orphan.json").write_text('{"members": [{"class_id":"a"}]}')
            with sqlite3.connect(root / "dataset/allocations.sqlite") as db:
                db.execute("CREATE TABLE members (class_id TEXT, split TEXT, bucket INTEGER)")
                db.executemany("INSERT INTO members VALUES (?,?,?)", [("a", "eval_standard", 1), ("b", "eval_standard", 1)])
            self.assertEqual(c.accepted_counts(root, "a"), {("eval_standard", 1): 1})

    def test_claim_uses_current_audited_policy_after_reservation_extension(self):
        record = {"allocation": {"job_dir": "/campaign/job", "key": "a", "jobid": "1"},
                  "directory": "/campaign/group", "group_id": "group"}
        cfg = {"started_at": 1, "reservations": "/superseded", "reservation_hashes": "/superseded-hashes"}
        current, hashes = {"revision": 5}, {"audit": "current"}
        with patch("scripts.campaign_20261110.acceptance.audit_job", return_value={}), \
             patch("scripts.campaign_20261110.goal_groups.load_template_contract", return_value="template"), \
             patch("scripts.campaign_20261110.group_lane.claim_group", return_value={"accepted": 2}) as claim, \
             patch("scripts.campaign_20261110.reservation_policy.active_policy", return_value=({}, current, hashes)), \
             patch.object(c, "write_json_atomic"):
            c.claim(record, {"class_id": "a"}, cfg, Path("/campaign"))
        self.assertEqual(claim.call_args.args[2], current)
        self.assertEqual(claim.call_args.kwargs["reservation_hashes"], hashes)
        self.assertEqual(record["status"], "claimed")


if __name__ == "__main__":
    unittest.main()
