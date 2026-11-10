import json
from pathlib import Path
import tempfile
import unittest

from scripts.campaign_20261110.infra_vast.campaign_inputs import prepare_environment, prepare_cache_environment, prepare_queue, prepare_pools, dispatch_order, runtime_scenes, assignment_plan, sha, QUEUE_ENV


class VastQueueInputsTests(unittest.TestCase):
    def test_parallel_plan_reserves_each_job_once_and_respects_worker_constraints(self):
        jobs = [{'id':str(i), 'gpu':'RTX 5090', 'time_s':1800} for i in range(5)]
        jobs[0]['avoid'] = 'ready'
        live = {name:{'gpu_name':gpu} for name,gpu in [('ready','RTX 5090'),('busy','RTX 5090'),
                ('boot','RTX 5090'),('old','RTX 5090'),('other','RTX 4090')]}
        heartbeats = {name:{'phase':'ready','slots':1} for name in live}
        heartbeats['boot']['phase'] = 'boot'
        created = {'old':{'time':0, 'max_lifetime_s':3000}}
        planned, remaining = assignment_plan(jobs, live, heartbeats, created, {'busy':['existing']}, 1000)
        self.assertEqual([(worker, job['id']) for worker,_,job in planned], [('ready','1')])
        self.assertEqual([job['id'] for job in remaining], ['0','2','3','4'])
        heartbeats['busy']['slots'] = 3
        planned, remaining = assignment_plan(jobs, live, heartbeats, created, {'busy':['existing']}, 1000)
        self.assertEqual([job['id'] for _,_,job in planned], ['1','0','2'])
        self.assertEqual([job['id'] for job in remaining], ['3','4'])

    def test_actual_scene_overrides_historical_counted_alias(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); config = root / 'config.json'
            config.write_text('{"template":{"scene":"Rs_int"}}')
            env = {'ISPATIALGYM_CAMPAIGN_CONFIG':str(config), 'TEMPLATE':str(root/'Rs_int__room-living_room_0'), 'TEMPLATE_ID':'house_single_floor__old'}
            self.assertEqual(runtime_scenes(env, root), ['Rs_int'])
            config.write_text('{"template":{"scene":"hotel_suite_large"}}')
            with self.assertRaisesRegex(ValueError, 'differs from actual source'):
                runtime_scenes(env, root)
    def test_warm_store_routing_preserves_native_key_selection_and_source_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); job, _, _ = self.fixture(root)
            env = job / 'job.env'
            env.write_text(env.read_text() + 'KIT_TPL_NFS_ROOT=' + str(root / 'kit_templates') + '\nKIT_TEX_STORE=' + str(root / 'kit_textures') + '\n')
            before = env.read_bytes()
            result = prepare_cache_environment(job, root)
            self.assertEqual((job / 'job.env.before_vast_caches').read_bytes(), before)
            self.assertIn('ispatialgym-lab-vec-eval-cache/kit_templates', env.read_text())
            self.assertEqual(result['selection'], 'native_exact_gpu_driver_behavior_complete_template')
            self.assertIsNone(prepare_cache_environment(job, root))

    def test_unexpected_cache_path_is_rejected_without_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); job, _, _ = self.fixture(root)
            env = job / 'job.env'; env.write_text('KIT_TPL_NFS_ROOT=/unreviewed/cache\n')
            before = env.read_bytes()
            with self.assertRaisesRegex(ValueError, 'unexpected cluster cache'):
                prepare_cache_environment(job, root)
            self.assertEqual(env.read_bytes(), before)
    def test_production_priority_never_bypasses_dispatch_allowlist(self):
        rows=[{"id":str(n),"status":"queued"} for n in (4,2,3,1)] + [{"id":"5","status":"assigned"}]
        self.assertEqual([r['id'] for r in dispatch_order(rows,{})],['1','2','3','4'])
        config={'dispatch_job_ids':['1','2','4','5'],'dispatch_priority_job_ids':['5','3','4']}
        self.assertEqual([r['id'] for r in dispatch_order(rows,config)],['4','1','2'])
        self.assertEqual(dispatch_order(rows,{'dispatch_job_ids':[]}),[])

    def test_pool_stays_at_exact_path_and_config_bytes_are_unchanged(self):
        for queued in (False, True):
            with self.subTest(queued=queued), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                job, queue, _ = self.fixture(root)
                config = root / "groups/g1/campaign_config.json"
                pool = root / "manifests/pool.json"
                pool.parent.mkdir()
                pool.write_text('{"schema":"isg-binding-pool-v1","candidate":"full_pool"}')
                config.write_text(json.dumps({"template":{"scene_setup":{"binding_pool":{
                    "schema":"isg-active-binding-pool-v1", "manifest":str(pool), "manifest_sha256":sha(pool)}}}}))
                before = config.read_bytes()
                env = {"ISPATIALGYM_CAMPAIGN_CONFIG":str(config)}
                if queued:
                    raw = json.loads(queue.read_text())
                    raw["config_sha256"] = sha(config)
                    queue.write_text(json.dumps(raw))
                    record = prepare_queue(job, {QUEUE_ENV:str(queue)}, root)
                    env = {QUEUE_ENV:record["staged_queue"]}
                self.assertEqual(prepare_pools(job, env, root), [str(pool)])
                self.assertEqual(config.read_bytes(), before)
                self.assertEqual(prepare_pools(job, env, root), [str(pool)])
                pool.write_text('{"schema":"isg-binding-pool-v1","candidate":"changed"}')
                with self.assertRaisesRegex(ValueError, "digest differs"):
                    prepare_pools(job, env, root)

    def test_external_pool_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "campaign"
            job, _, _ = self.fixture(root)
            config = root / "groups/g1/campaign_config.json"
            pool = Path(tmp) / "outside.json"
            pool.write_text('{"schema":"isg-binding-pool-v1"}')
            config.write_text(json.dumps({"template":{"scene_setup":{"binding_pool":{
                "schema":"isg-active-binding-pool-v1", "manifest":str(pool), "manifest_sha256":sha(pool)}}}}))
            with self.assertRaisesRegex(ValueError, "outside owned"):
                prepare_pools(job, {"ISPATIALGYM_CAMPAIGN_CONFIG":str(config)}, root)
            self.assertFalse((job / "campaign_pool_inputs.json").exists())

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
