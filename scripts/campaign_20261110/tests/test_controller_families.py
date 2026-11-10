from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from scripts.campaign_20261110 import controller_families as c


class FamilyControllerTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
