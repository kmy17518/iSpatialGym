import json
from pathlib import Path
import tempfile
import unittest

from scripts.campaign_20261110.infra_vast.campaign_inputs import prepare_environment, prepare_queue, sha, QUEUE_ENV


class VastQueueInputsTests(unittest.TestCase):
    def test_cluster_apps_override_removed_with_original_environment_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job, _, _ = self.fixture(root)
            env = job / "job.env"
            env.write_text(env.read_text() + "ISGF_EXP_PATH=" + str(root / "isaacsim_overlay/apps") + "\n")
            before = env.read_bytes()
            record = prepare_environment(job, root)
            self.assertEqual((job / "job.env.before_vast_apps").read_bytes(), before)
            self.assertNotIn("ISGF_EXP_PATH", env.read_text())
            self.assertEqual(record["after_sha256"], sha(env))
            self.assertIsNone(prepare_environment(job, root))

    def test_unexpected_apps_path_fails_before_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job, _, _ = self.fixture(root)
            env = job / "job.env"
            env.write_text("ISGF_EXP_PATH=/unreviewed/apps\n")
            with self.assertRaisesRegex(ValueError, "unexpected Isaac"):
                prepare_environment(job, root)
            self.assertEqual(env.read_text(), "ISGF_EXP_PATH=/unreviewed/apps\n")

    def fixture(self, root, schema="isg-held-goal-queue-v1"):
        job = root / "out/queue/g1"
        group = root / "groups/g1"
        queue = root / "queues/q1/queue.json"
        for path in (job, group, queue.parent):
            path.mkdir(parents=True)
        config = group / "campaign_config.json"
        config.write_text('{"campaign_id":"campaign_20261110"}')
        plan = group / "goal_plan.json"
        plan.write_text('{"campaign_id":"campaign_20261110","group_id":"g1","max_members":3}')
        queue.write_text(json.dumps({"schema":schema, "campaign_id":"campaign_20261110",
                                    "config_path":str(config), "config_sha256":sha(config),
                                    "entries":[{"path":str(plan), "sha256":sha(plan)}]}))
        (job / "job.env").write_text(QUEUE_ENV + "=" + str(queue) + "\nOTHER=preserved\n")
        return job, queue, plan

    def test_inputs_and_runtime_journal_are_inside_returned_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job, queue, _ = self.fixture(root)
            before = sha(queue)
            record = prepare_queue(job, {QUEUE_ENV:str(queue)}, root)
            self.assertEqual(sha(queue), before)
            self.assertTrue(Path(record["journal"]).is_relative_to(job))
            raw = json.loads(Path(record["staged_queue"]).read_text())
            self.assertTrue(Path(raw["entries"][0]["path"]).is_file())
            self.assertIn("OTHER=preserved", (job / "job.env").read_text())
            self.assertEqual(prepare_queue(job, {QUEUE_ENV:record["staged_queue"]}, root), record)

    def test_both_native_queue_protocols_keep_exact_plan_bytes(self):
        for schema in ("isg-held-goal-queue-v1", "isg-pair-goal-queue-v1"):
            with self.subTest(schema=schema), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                job, queue, plan = self.fixture(root, schema)
                before = sha(plan)
                record = prepare_queue(job, {QUEUE_ENV:str(queue)}, root)
                copied = json.loads(Path(record["staged_queue"]).read_text())
                self.assertEqual(copied["schema"], schema)
                self.assertEqual(sha(copied["entries"][0]["path"]), before)

    def test_tampered_plan_fails_before_environment_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job, queue, plan = self.fixture(root)
            before = (job / "job.env").read_bytes()
            plan.write_text("changed")
            with self.assertRaisesRegex(ValueError, "digest differs"):
                prepare_queue(job, {QUEUE_ENV:str(queue)}, root)
            self.assertEqual((job / "job.env").read_bytes(), before)

    def test_external_input_is_not_shipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "campaign"
            job, queue, _ = self.fixture(root)
            outside = Path(tmp) / "outside.json"
            outside.write_text("{}")
            raw = json.loads(queue.read_text())
            raw["config_path"], raw["config_sha256"] = str(outside), sha(outside)
            queue.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "outside owned"):
                prepare_queue(job, {QUEUE_ENV:str(queue)}, root)


if __name__ == "__main__":
    unittest.main()
