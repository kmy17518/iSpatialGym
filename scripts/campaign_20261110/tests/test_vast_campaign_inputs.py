import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.campaign_20261110.infra_vast.campaign_inputs import prepare_environment, prepare_cache_environment, prepare_queue, prepare_pools, dispatch_order, runtime_scenes, assignment_plan, staging_record, sha, QUEUE_ENV
from scripts.campaign_20261110.infra_vast.campaign_inputs import compatible_gpus, uncovered_jobs, worker_can_run
from scripts.campaign_20261110.infra_vast.campaign_inputs import QUEUE_SCHEMAS
from scripts.campaign_20261110.infra_vast.campaign_inputs import drained_workers, rental_accounts, drained_worker_finished


class VastQueueInputsTests(unittest.TestCase):
    def test_idle_worker_is_not_retained_for_jobs_that_cannot_finish_before_lease_expiry(self):
        worker={'gpu_name':'RTX 4090'}
        creation={'time':0,'max_lifetime_s':21600}
        queued={'id':'long','gpu':'RTX 4090','time_s':16200}
        self.assertTrue(worker_can_run(queued,'warm',worker,creation,4500))
        self.assertFalse(worker_can_run(queued,'warm',worker,creation,4501))
        # This warm card still serves shorter tasks; it is not retired just for age.
        self.assertTrue(worker_can_run({**queued,'time_s':1800},'warm',worker,creation,14000))
        self.assertFalse(worker_can_run({**queued,'time_s':1800,'avoid':'warm'},'warm',worker,creation,14000))
        self.assertEqual(uncovered_jobs([queued],{'warm':worker},{'warm':{'phase':'ready','slots':1}},
                                        {'warm':creation},{},4501),[queued])

    def test_account_drain_preserves_busy_workers_and_routes_new_work_to_funded_account(self):
        config={'drain_accounts':['secondary'], 'rental_accounts':['primary']}
        live={'one':{'_key_name':'primary','gpu_name':'RTX 4090'},
              'two':{'_key_name':'secondary','gpu_name':'RTX 4090'}}
        draining=drained_workers(live,config)
        self.assertEqual(draining,{'two'})
        self.assertEqual(rental_accounts(('primary','secondary'),config,{'primary':20}),['primary'])
        self.assertFalse(drained_worker_finished('two',draining,{'two':['job']},{}))
        self.assertFalse(drained_worker_finished('two',draining,{}, {'running':{'job':{}}}))
        self.assertTrue(drained_worker_finished('two',draining,{}, {'phase':'ready'}))
        self.assertFalse(drained_worker_finished('one',draining,{},{}))
        jobs=[{'id':'j','gpu':'RTX 4090','time_s':1800}]
        planned,_=assignment_plan(jobs,{k:v for k,v in live.items() if k not in draining},
                                  {'one':{'phase':'ready','slots':1},'two':{'phase':'ready','slots':1}}, {},{},100)
        self.assertEqual([worker for worker,_,_ in planned],['one'])
    def test_proven_gpu_substitution_uses_idle_worker_and_does_not_double_count_boot_capacity(self):
        config = {'gpus':['RTX 4090','RTX 5090'],'allow_proven_gpu_substitution':True,'vram_per_job_gb':20,'max_jobs_per_instance':1}
        jobs = [{'id':str(i),'gpu':'RTX 4090','time_s':1800,'compatible_gpus':compatible_gpus('RTX 4090',config)} for i in range(3)]
        live={'one':{'gpu_name':'RTX 5090'},'two':{'gpu_name':'RTX 5090'}}
        hb={'one':{'phase':'ready','slots':1},'two':{'phase':'boot'}}
        planned,remaining=assignment_plan(jobs,live,hb,{}, {},1000)
        self.assertEqual([j['id'] for _,_,j in planned],['0'])
        self.assertEqual([j['id'] for j in uncovered_jobs(remaining,live,hb,{}, {'one':['0']},1000)],['2'])
        self.assertEqual(jobs[0]['gpu'],'RTX 4090')
        for changes in [{'vram_per_job_gb':24},{'max_jobs_per_instance':2},{'allow_proven_gpu_substitution':False}]:
            self.assertEqual(compatible_gpus('RTX 4090',{**config,**changes}),['RTX 4090'])
        self.assertEqual(compatible_gpus('L40S',config),['L40S'])
    def test_staging_reads_only_target_state_without_ingesting_or_mutating(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root/'900001.json'; target.write_text('{"id":"900001","status":"queued"}')
            (root/'900002.json').write_text('unrelated partially published external input')
            before = target.read_bytes()
            value = staging_record(root, '900001'); value['status'] = 'assigned'
            self.assertEqual(target.read_bytes(), before)
            with self.assertRaises(FileNotFoundError): staging_record(root, '900003')
            with self.assertRaises(ValueError): staging_record(root, '../900001')
            target.write_text('{"id":"900002"}')
            with self.assertRaisesRegex(ValueError, 'identity'): staging_record(root, '900001')

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

    def test_relocated_template_uses_native_metadata_scene_and_still_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);template=root/'templates/alignment-world_vertical_02';template.mkdir(parents=True)
            (template/'template_meta.json').write_text('{"scene":"house_double_floor_upper"}')
            config=root/'config.json';config.write_text('{"template":{"scene":"house_double_floor_upper"}}')
            env={'TEMPLATE':str(template),'ISPATIALGYM_CAMPAIGN_CONFIG':str(config)}
            self.assertEqual(runtime_scenes(env,root),['house_double_floor_upper'])
            config.write_text('{"template":{"scene":"Rs_int"}}')
            with self.assertRaisesRegex(ValueError,'differs from actual source'):runtime_scenes(env,root)
            conventional=root/'Rs_int__room-living';template.rename(conventional);env['TEMPLATE']=str(conventional)
            with self.assertRaisesRegex(ValueError,'conventional scene directory differ'):runtime_scenes(env,root)
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
    @patch("scripts.campaign_20261110.infra_vast.campaign_inputs.production_decision", return_value={"allowed":True})
    def test_production_priority_never_bypasses_dispatch_allowlist(self, gate):
        rows=[{"id":str(n),"status":"queued"} for n in (4,2,3,1)] + [{"id":"5","status":"assigned"}]
        self.assertEqual([r['id'] for r in dispatch_order(rows,{})],['1','2','3','4'])
        config={'dispatch_job_ids':['1','2','4','5'],'dispatch_priority_job_ids':['5','3','4']}
        self.assertEqual([r['id'] for r in dispatch_order(rows,config)],['4','1','2'])
        self.assertEqual(dispatch_order(rows,{'dispatch_job_ids':[]}),[])

    def test_dispatch_gate_blocks_train_even_with_priority_and_missing_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); rows=[]
            for n,split in enumerate(('train','eval_standard','eval_model')):
                job=root/str(n);job.mkdir();config=job/'campaign_config.json'
                config.write_text(json.dumps({'template':{'split':split,'task':'test','class_id':'test'}}))
                (job/'job.env').write_text('ISPATIALGYM_CAMPAIGN_CONFIG='+str(config)+'\n')
                rows.append({'id':str(n),'status':'queued','job_dir':str(job)})
            rows.append({'id':'3','status':'queued','job_dir':str(root/'missing')})
            config={'campaign_root':str(root),'dispatch_priority_job_ids':['0']}
            self.assertEqual([j['id'] for j in dispatch_order(rows,config)],['1','2'])
            # Editing the policy to a nominal open flag cannot authorize train.
            (root/'controllers').mkdir();(root/'controllers/training_production_gate.json').write_text(json.dumps(
                {'schema':'isg-training-production-gate-v1','campaign_id':'campaign_20261110','mode':'open'}))
            self.assertEqual([j['id'] for j in dispatch_order(rows,config)],['1','2'])

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
        for schema in QUEUE_SCHEMAS:
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
def test_template_customs_dependency_keeps_original_cross_revision_bytes(tmp_path):
    from scripts.campaign_20261110.infra_vast.campaign_inputs import template_customs_inputs
    import hashlib
    import json
    template = tmp_path / 'out/old/inst/task/scene'
    template.mkdir(parents=True)
    customs = tmp_path / 'code/old/assets/isg_customs.json'
    customs.parent.mkdir(parents=True)
    customs.write_text('{"task": {"roles": {}}}\n')
    metadata = template / 'template_meta.json'
    metadata.write_text(json.dumps({'customs': str(customs), 'variant': 'task'}))
    before = metadata.read_bytes()
    assert template_customs_inputs({'TEMPLATE': str(template), 'ISGE_CODE': str(tmp_path / 'code/new')}, tmp_path) == [
        {'path': str(customs), 'sha256': hashlib.sha256(customs.read_bytes()).hexdigest()}]
    assert metadata.read_bytes() == before


def test_template_customs_dependency_refuses_external_source(tmp_path):
    from scripts.campaign_20261110.infra_vast.campaign_inputs import template_customs_inputs
    import json
    import pytest
    root = tmp_path / 'campaign'
    template = root / 'out/template'
    template.mkdir(parents=True)
    outside = tmp_path / 'unowned.json'
    outside.write_text('{}')
    (template / 'template_meta.json').write_text(json.dumps({'customs': str(outside)}))
    with pytest.raises(ValueError, match='outside owned campaign'):
        template_customs_inputs({'TEMPLATE': str(template)}, root)


def test_exact_external_customs_requires_both_source_and_template_hashes(tmp_path):
    import pytest
    from scripts.campaign_20261110.infra_vast.campaign_inputs import template_customs_inputs
    root=tmp_path/'campaign';template=root/'templates/task/instance';template.mkdir(parents=True)
    external=tmp_path/'original_customs.json';external.write_text('{"original":"approved"}')
    metadata=template/'template_meta.json';metadata.write_text(json.dumps({'customs':str(external)}))
    receipt=root/'manifests/external_template_customs_inputs.json';receipt.parent.mkdir()
    receipt.write_text(json.dumps({'schema':'isg-exact-external-template-customs-v1','campaign_id':'campaign_20261110',
        'sources':{str(external):{'sha256':sha(external),'templates':{str(template):sha(metadata)}}}}))
    assert template_customs_inputs({'TEMPLATE':str(template)},root)==[{'path':str(external),'sha256':sha(external)}]
    metadata.write_text(json.dumps({'customs':str(external),'changed':True}))
    with pytest.raises(ValueError,match='identity differ'):template_customs_inputs({'TEMPLATE':str(template)},root)
    metadata.write_text(json.dumps({'customs':str(external)}));external.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='identity differ'):template_customs_inputs({'TEMPLATE':str(template)},root)


def test_camera_context_stages_checked_source_proof_and_review_image_closure(tmp_path):
    from dataclasses import asdict
    import json
    import pytest
    from scripts.campaign_20261110.tests.test_camera_context import fixture
    from scripts.campaign_20261110.camera_context import bind_camera_context
    from scripts.campaign_20261110.infra_vast.campaign_inputs import camera_context_inputs
    template, args = fixture(tmp_path)
    bound = bind_camera_context(template, **args)
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'template': asdict(bound)}))
    env = {'ISPATIALGYM_CAMPAIGN_CONFIG': str(config)}
    rows = camera_context_inputs(env, tmp_path)
    assert {Path(row['path']).name for row in rows} == {
        'full_scene.json', 'context_probe.json', 'table_scene.json', 'view.png', 'support_probe.json', 'visual.json'}
    assert not any('.encrypted.usd' in row['path'] for row in rows)
    (tmp_path / 'proofs/view.png').write_bytes(b'changed')
    with pytest.raises(ValueError, match='bytes changed'):
        camera_context_inputs(env, tmp_path)



def test_native_handle_closure_uses_template_variant_not_counted_alias(tmp_path):
    from scripts.campaign_20261110.infra_vast.campaign_inputs import template_customs_inputs
    root=tmp_path/'campaign';template=root/'templates/wardrobe';template.mkdir(parents=True)
    handle=root/'private/handles.json';handle.parent.mkdir();handle.write_text('{"native":"exact"}')
    customs=root/'customs.json';customs.write_text(json.dumps({
        'wardrobe':{'roles':{'target':{'handle_file':{'scene/object':str(handle)}}}},
        'dishwasher':{'roles':{'target':{'handle_file':'/unrelated/absent.json'}}}}))
    meta=template/'template_meta.json';meta.write_text(json.dumps({'customs':str(customs),'variant':'wardrobe'}))
    before=meta.read_bytes()
    result=template_customs_inputs({'TEMPLATE':str(template),'ALIAS':'dishwasher'},root)
    assert result==[{'path':str(customs),'sha256':sha(customs)},{'path':str(handle),'sha256':sha(handle)}]
    assert meta.read_bytes()==before
    handle.unlink()
    import pytest
    with pytest.raises(ValueError,match='native handle dependency missing'):
        template_customs_inputs({'TEMPLATE':str(template)},root)


def test_native_handle_closure_refuses_external_absolute_files(tmp_path):
    import pytest
    from scripts.campaign_20261110.infra_vast.campaign_inputs import template_customs_inputs
    root=tmp_path/'campaign';template=root/'templates/wardrobe';template.mkdir(parents=True)
    outside=tmp_path/'outside.json';outside.write_text('{}')
    customs=root/'customs.json';customs.write_text(json.dumps({'wardrobe':{'roles':{'target':{'handle_file':str(outside)}}}}))
    (template/'template_meta.json').write_text(json.dumps({'customs':str(customs),'variant':'wardrobe'}))
    with pytest.raises(ValueError,match='outside owned campaign'):
        template_customs_inputs({'TEMPLATE':str(template)},root)


def test_standard_recovery_precedes_training_without_enabling_parked_jobs(monkeypatch):
    from scripts.campaign_20261110.infra_vast import campaign_inputs
    monkeypatch.setattr(campaign_inputs, 'production_decision', lambda *args: {'allowed':True})
    from scripts.campaign_20261110.infra_vast import cost_provenance
    evidence={
        '1':{'split':'train','class_id':'zero_class'},
        '2':{'split':'eval_standard','class_id':'working_class'},
        '3':{'split':'eval_standard','class_id':'zero_class'},
        '4':{'split':'eval_model','class_id':'zero_class'},
        '5':{'split':'eval_standard','class_id':'zero_class'},
        '6':{'split':None,'class_id':'zero_class'},
    }
    monkeypatch.setattr(cost_provenance,'scope',lambda row:{'status':'observed',**evidence[row['id']]})
    rows=[{'id':str(i),'status':'queued'} for i in range(1,7)]
    config={'dispatch_job_ids':['1','2','3','4','6'], 'dispatch_priority_job_ids':['1','4','2'],
            'prioritize_standard_eval':True,'standard_recovery_class_ids':['zero_class']}
    assert [r['id'] for r in dispatch_order(rows,config)]==['3','2','4','6','1']
    # An explicitly reviewed prerequisite can advance without guessing its split.
    config['standard_recovery_priority_job_ids']=['6']
    assert [r['id'] for r in dispatch_order(rows,config)]==['3','6','2','4','1']


def exact_customs_preflight_fixture(tmp_path):
    from dataclasses import asdict
    from scripts.campaign_20261110.contracts import ObjectRef, Template
    root=tmp_path/'campaign';template=root/'templates/task';template.mkdir(parents=True)
    external=tmp_path/'frozen_customs.json';external.write_text('{"task":{"roles":{}}}')
    (template/'template.json').write_text('{"exact":"source"}')
    bddl=template/'task.bddl';bddl.write_text('exact original definition')
    meta=template/'template_meta.json';meta.write_text(json.dumps({'variant':'task','customs':str(external),
        'activity_definition':'task.bddl','bddl_sha256':sha(bddl)}))
    obj=ObjectRef('target','target','cube','ready','a'*64,ready=True,readiness_source='native proof')
    t=Template('task','mental_rotation','mental_rotation-toy_figure','eval_standard','house_single_floor','room',(obj,),
               'b'*64,{'source_template':str(template),'source_template_sha256':sha(template/'template.json'),
                        'source_bddl_sha256':sha(bddl)})
    cfg=root/'config.json';cfg.write_text(json.dumps({'campaign_id':'campaign_20261110','purpose':'validation',
        'split':'eval_standard','template':asdict(t),'validation_key':{'code_sha256':'c'*64}}))
    receipt=root/'manifests/external_template_customs_inputs.json';receipt.parent.mkdir()
    receipt.write_text(json.dumps({'schema':'isg-exact-external-template-customs-v1','campaign_id':'campaign_20261110',
        'sources':{str(external):{'sha256':sha(external),'templates':{}}}}))
    return root, {'TEMPLATE':str(template),'ISPATIALGYM_CAMPAIGN_CONFIG':str(cfg)}, receipt, external


def test_exact_customs_preflight_joins_new_ready_config_without_mutating_frozen_source(tmp_path):
    from scripts.campaign_20261110.infra_vast.campaign_inputs import register_exact_template_customs_inputs,template_customs_inputs
    root,env,receipt,external=exact_customs_preflight_fixture(tmp_path)
    before=external.read_bytes();joins=register_exact_template_customs_inputs(env,root)
    assert len(joins)==1 and external.read_bytes()==before
    assert template_customs_inputs(env,root)==[{'path':str(external),'sha256':sha(external)}]
    recorded=receipt.read_bytes();assert register_exact_template_customs_inputs(env,root)==[]
    assert receipt.read_bytes()==recorded


def test_exact_customs_preflight_rejects_wrong_source_unready_and_changed_bytes(tmp_path):
    from scripts.campaign_20261110.infra_vast.campaign_inputs import register_exact_template_customs_inputs
    import pytest
    root,env,receipt,external=exact_customs_preflight_fixture(tmp_path);before=receipt.read_bytes()
    cfg=Path(env['ISPATIALGYM_CAMPAIGN_CONFIG']);original=cfg.read_bytes()
    value=json.loads(original);value['template']['scene_setup']['source_template']=str(root/'other')
    cfg.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='source template differs'):register_exact_template_customs_inputs(env,root)
    value=json.loads(original);value['template']['objects'][0]['ready']=False;cfg.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='confirmed readiness'):register_exact_template_customs_inputs(env,root)
    cfg.write_bytes(original);external.write_text('{"task":{"changed":true}}')
    with pytest.raises(ValueError,match='previously frozen'):register_exact_template_customs_inputs(env,root)
    assert receipt.read_bytes()==before


def test_exact_customs_preflight_cannot_introduce_new_external_source(tmp_path):
    from scripts.campaign_20261110.infra_vast.campaign_inputs import register_exact_template_customs_inputs
    import pytest
    root,env,receipt,external=exact_customs_preflight_fixture(tmp_path)
    doc=json.loads(receipt.read_text());doc['sources']={};receipt.write_text(json.dumps(doc));before=receipt.read_bytes()
    with pytest.raises(ValueError,match='previously frozen'):register_exact_template_customs_inputs(env,root)
    assert receipt.read_bytes()==before


def test_exact_customs_preflight_supports_direct_stage_entrypoint_from_other_cwd(tmp_path):
    import subprocess, sys
    from scripts.campaign_20261110.infra_vast import campaign_inputs
    root,env,receipt,external=exact_customs_preflight_fixture(tmp_path)
    command="import runpy,json; module=runpy.run_path("+repr(campaign_inputs.__file__)+"); module['register_exact_template_customs_inputs']("+repr(env)+","+repr(str(root))+")"
    result=subprocess.run([sys.executable,'-I','-c',command],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(receipt.read_text())['automatic_exact_source_joins']
