from pathlib import Path
import json
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from scripts.campaign_20261110 import controller_families as c


class FamilyControllerTests(unittest.TestCase):
    def test_terminal_group_reuses_only_exact_source_key_and_keeps_runtime_failure_evidence(self):
        from scripts.campaign_20261110.tests.test_acceptance import fixture
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root,prefix='source_');target=fixture(root,prefix='target_')
            directory=root/'groups/target_group';directory.mkdir(parents=True)
            block=json.loads(next((target.job/'inst').rglob('*-tro_state.json')).read_text())['ispatialgym']['campaign']
            config={key:block[key] for key in ('validation_key','campaign_id','split')}
            c.write_json_atomic(directory/'campaign_config.json',config)
            source_row={'key':'source','jobid':'1','job_dir':str(source.job)}
            gate=c.gate_path(root,source_row);gate.parent.mkdir(parents=True)
            # Keep the real fixture replay under the relocated gate's parent.
            import shutil
            shutil.copytree(source.gate.parent/'replay',gate.parent/'replay')
            gate.write_bytes(source.gate.read_bytes())
            row={'key':'target','jobid':'2','job_dir':str(target.job)}
            record={'allocation':row,'directory':str(directory),'group_id':'target_group'}
            anchor={'allocation':source_row};cfg={'started_at':source.since}
            self.assertIsNone(c.reusable_group_gate(record,anchor,cfg,root,job_ended=False))
            (target.job/'sample.log').write_text('CUDA error: an illegal memory access was encountered\n')
            self.assertEqual(c.reusable_group_gate(record,anchor,cfg,root,job_ended=True),gate)
            self.assertIn('terminal_runtime_failure',record)
            self.assertEqual(record['release_gate']['sha256'],c.file_digest(gate))
            from scripts.campaign_20261110.acceptance import audit_job
            reviewed=audit_job(target.job,gate,root=root,started_at=source.since,group_closed=True,group_id='target_group')
            c.write_json_atomic(directory/'release_review.json',reviewed)
            record.pop('release_gate')
            self.assertEqual(c.reusable_group_gate(record,anchor,cfg,root,job_ended=True),gate)
            # Changed current member QA cannot be hidden by a valid source gate.
            report=json.loads(target.reports[0].read_text());report['qa']['enabled']=False
            c.write_json_atomic(target.reports[0],report)
            with self.assertRaisesRegex(ValueError,'QA'):
                c.reusable_group_gate(record,anchor,cfg,root,job_ended=True)
            report['qa']['enabled']=True;c.write_json_atomic(target.reports[0],report)
            (directory/'release_review.json').unlink()
            record.pop('release_gate');original_physics=config['validation_key']['physics_sha256'];config['validation_key']['physics_sha256']='d'*64
            c.write_json_atomic(directory/'campaign_config.json',config)
            self.assertIsNone(c.reusable_group_gate(record,anchor,cfg,root,job_ended=True))
            config['validation_key']['physics_sha256']=original_physics;c.write_json_atomic(directory/'campaign_config.json',config)
            c.write_json_atomic(c.gate_path(root,row),{'success':False})
            self.assertIsNone(c.reusable_group_gate(record,anchor,cfg,root,job_ended=True))

    def migration_fixture(self, root):
        old=root/'groups/old';new=root/'groups/new'
        for folder in (old,new):
            folder.mkdir(parents=True);(folder/'campaign_config.json').write_text('{"key":"same"}')
        (new/'goal_plan.json').write_text('{"group_id":"new"}')
        allocation={'jobid':'12','n':'3','key':'a','code':'/snapshot','tag':'goal_groups'}
        proof={'schema':'isg-never-started-migration-v1','never_started':True,'original_allocation':allocation,
            'after':{'accounting':{'state':'CANCELLED','start':'Unknown'}},
            'final_execution_proof':{'nonempty_execution_artifacts':[]}}
        proof_path=root/'migration.json';proof_path.write_text(json.dumps(proof))
        ready=root/'dispatch/ready/new.json';ready.parent.mkdir(parents=True)
        command=['python','submit_fast.py','submit','a','3']
        ready.write_text(json.dumps({'command':command,'command_sha256':c.digest(command),
            'identity':{'start':50000100,'n':3,'code':'/snapshot','tag':'goal_groups','key':'a'}}))
        receipt=root/'dispatch/receipts/old.json';receipt.parent.mkdir(parents=True)
        migration={'never_started':True,'proof':{'path':str(proof_path),'sha256':c.file_digest(proof_path)},
            'replacement':{'directory':str(new),'index':50000100,'count':3,'dispatch_intent':str(ready),
                'dispatch_receipt':str(receipt.with_name('new.json')),'command':command,'group_id':'new'}}
        receipt.write_text(json.dumps({'allocation':allocation,'migration':migration}))
        stage={'group_id':'old','directory':str(old),'index':50000000,'status':'waiting',
               'allocation':allocation,'dispatch_receipt':str(receipt)}
        return stage,proof_path,receipt,new

    def test_migration_adopts_only_verified_fresh_replacement_and_preserves_prior_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stage,_,_,_=self.migration_fixture(root)
            self.assertEqual(c.adopt_pending_replacement(stage,root),'adopted')
            self.assertEqual(stage['group_id'],'new');self.assertEqual(stage['index'],50000100)
            self.assertNotIn('allocation',stage)
            self.assertEqual(stage['status'],'external_submission')
            self.assertEqual(stage['migration_history'][0]['prior_stage']['allocation']['jobid'],'12')
            self.assertIsNone(c.adopt_pending_replacement(stage,root))

    def test_migration_waits_for_cancellation_and_rejects_changed_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stage,_,receipt,new=self.migration_fixture(root)
            data=json.loads(receipt.read_text());receipt.write_text(json.dumps({'migration_pending':{'request':'queued'}}))
            self.assertEqual(c.adopt_pending_replacement(stage,root),'pending')
            self.assertEqual(stage['allocation']['jobid'],'12')
            receipt.write_text(json.dumps(data));(new/'campaign_config.json').write_text('{"key":"changed"}')
            with self.assertRaisesRegex(ValueError,'exact collection configuration'):c.adopt_pending_replacement(stage,root)
            self.assertEqual(stage['index'],50000000)

    def test_migration_rejects_a_previously_started_job_even_with_valid_proof_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stage,proof,receipt,_=self.migration_fixture(root)
            data=json.loads(proof.read_text());data['after']['accounting']['start']='2026-11-10T00:00:00'
            proof.write_text(json.dumps(data));data=json.loads(receipt.read_text())
            data['migration']['proof']['sha256']=c.file_digest(proof);receipt.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'may have executed'):c.adopt_pending_replacement(stage,root)

    def test_reuse_requires_opt_in_exact_runtime_and_genuine_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);state,path=self.recovered_fixture(root)
            state['history'][0]['name']='source_cycle'
            state['history'][0]['anchor']['allocation']['code']='/snapshot'
            spec={'class_id':'a','anchor_snapshot':'/snapshot'}
            self.assertIsNone(c.reusable_anchor_record(spec,state,root))
            spec['reuse_successful_anchor']=True
            record=c.reusable_anchor_record(spec,state,root)
            self.assertEqual(record['status'],'waiting')
            self.assertEqual(record['reused_from_cycle'],'source_cycle')
            self.assertEqual(state['history'][0]['anchor']['status'],'failed')
            spec['anchor_snapshot']='/other'
            self.assertIsNone(c.reusable_anchor_record(spec,state,root))
            spec['anchor_snapshot']='/snapshot'
            gate=json.loads(path.read_text());gate['replay_mode']='kinematic';path.write_text(json.dumps(gate))
            self.assertIsNone(c.reusable_anchor_record(spec,state,root))

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

    def test_proven_bucket_can_continue_without_claiming_other_buckets_complete(self):
        self.assertEqual(c.next_bucket({('eval_standard',0):1},[5,5],[0]),0)
        counts={('eval_standard',0):5}
        self.assertIsNone(c.next_bucket(counts,[5,5],[0]))
        self.assertEqual(c.next_bucket(counts,[5,5]),1)
        state={}
        with patch.object(c,'accepted_counts',return_value=counts):
            c.advance({'class_id':'class','eval_quotas':[5,5],'enabled_buckets':[0]},state,{},Path('/campaign'),{},1,lambda:None)
        self.assertEqual(state['status'],'validated_buckets_complete_other_buckets_need_diagnosis')
        for invalid in ([],[2],[-1],[0,0],[True]):
            with self.assertRaises(ValueError):c.next_bucket({},[5,5],invalid)

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
