import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from scripts.campaign_20261110.runtime_validation_metrics import attempts, accounting, rates, register, reference, validate_row, json_lines, sample_distribution, vast_accounting, member_timing


class MetricsTest(unittest.TestCase):
    def test_additive_destination_preserves_existing_measurements(self):
        from unittest.mock import patch
        from scripts.campaign_20261110.runtime_validation_metrics import observe
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);run=root/'runtime_validation/run';existing=run/'measurements'
            existing.mkdir(parents=True);(existing/'current.json').write_text('original')
            destination=root/'reviews/new'
            with patch('scripts.campaign_20261110.runtime_suite_paths.all_rows',return_value=[]),patch('scripts.campaign_20261110.runtime_validation_metrics.scheduler',return_value=({}, {'jobs':[]})):
                result=observe(run,destination)
            self.assertEqual((existing/'current.json').read_text(),'original')
            self.assertEqual(list(existing.iterdir()),[existing/'current.json'])
            self.assertEqual(json.loads((destination/'current.json').read_text()),result)
            self.assertTrue((destination/'current.md').exists())
            self.assertTrue(Path(result['scheduler_evidence']['path']).is_relative_to(destination))

    def fixture(self, root, camera=False):
        def put(path, value):
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value)); return reference(path)
        native=root/'code/trunk_exact';native.mkdir(parents=True);(native/'.snapshot_commit').write_text('abc123\n')
        cfg=put(root/'config.json', {'campaign_id':'campaign_20261110', 'template':{'class_id':'camera_a'if camera else'class_a', 'task':'camera_relocalization'if camera else'disembedding'}, 'validation_key':{'code_sha256':reference(native/'.snapshot_commit')['sha256']}})
        plan=put(root/'plan.json', {'group_id':'g1','max_members':3})
        row={'run_id':'one','run_started_at':1,'class_id':'camera_a'if camera else'class_a','frozen_runtime':{'path':str(native),'snapshot_sha256':reference(native/'.snapshot_commit')['sha256'],'commit':'abc123'},'campaign_config':cfg,'fresh_start':100,'fresh_count':1 if camera else 3,'camera_exempt':camera}
        if not camera:row.update(group_id='g1',goal_plan=plan)
        path=root/'suite/rows'/(row['class_id']+'.json');put(path,row);return path,row

    def event(self, row, ident='a', event='started', t=1, index=100, **kwargs):
        return {'class_id':row['class_id'],'config_sha256':row['campaign_config']['sha256'],'validation_key':{'code_sha256':row['frozen_runtime']['snapshot_sha256']},'goal_group_id':row.get('group_id'),'index':index,'attempt_id':ident,'event':event,'created_at':t,**kwargs}

    def test_registration_is_idempotent_and_immutable(self):
        with tempfile.TemporaryDirectory()as t:
            path,row=self.fixture(Path(t));first=register(path);self.assertEqual(first,register(path))
            row['fresh_start']=200;path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError,'registered suite row changed'):register(path)

    def test_runtime_and_config_must_match(self):
        with tempfile.TemporaryDirectory()as t:
            _,row=self.fixture(Path(t));cfg=json.loads(Path(row['campaign_config']['path']).read_text());cfg['validation_key']['code_sha256']='bad';Path(row['campaign_config']['path']).write_text(json.dumps(cfg));row['campaign_config']=reference(row['campaign_config']['path'])
            with self.assertRaisesRegex(ValueError,'suite runtime'):validate_row(row)

    def test_camera_has_no_group_denominator(self):
        with tempfile.TemporaryDirectory()as t:
            _,row=self.fixture(Path(t),True);validate_row(row);row['group_id']='invented'
            with self.assertRaisesRegex(ValueError,'fabricated'):validate_row(row)

    def test_requested_slots_are_not_actual_draws(self):
        with tempfile.TemporaryDirectory()as t:
            _,row=self.fixture(Path(t));events=[self.event(row),self.event(row,event='finished',t=5,outcome='rejected',reason='no_ik'),self.event(row,ident='b',t=6)]
            out=attempts(events,row);self.assertEqual(out['attempted_instance_slots'],1);self.assertEqual(out['native_draws'],2);self.assertEqual(out['censored_native_draws'],1);self.assertEqual(out['failed_native_process_seconds'],4)

    def test_duplicate_journal_reads_do_not_double_count(self):
        with tempfile.TemporaryDirectory()as t:
            _,row=self.fixture(Path(t));e=self.event(row);self.assertEqual(attempts([e,e],row)['native_draws'],1)

    def test_foreign_attempt_rejected(self):
        with tempfile.TemporaryDirectory()as t:
            _,row=self.fixture(Path(t));e=self.event(row);e['config_sha256']='foreign'
            with self.assertRaisesRegex(ValueError,'config'):attempts([e],row)

    def test_unmatched_finish_is_not_a_completed_draw(self):
        with tempfile.TemporaryDirectory()as t:
            _,row=self.fixture(Path(t))
            with self.assertRaisesRegex(ValueError,'durable start'):attempts([self.event(row,event='finished')],row)

    def test_native_success_milestones_and_censored_start_are_separate(self):
        with tempfile.TemporaryDirectory()as t:
            _,row=self.fixture(Path(t))
            events=[self.event(row,t=10),self.event(row,event='finished',t=30,outcome='written'),
                    self.event(row,ident='b',index=101,t=31),self.event(row,ident='b',index=101,event='finished',t=70,outcome='written'),
                    self.event(row,ident='c',index=102,t=71)]
            result=attempts(events,row)
            self.assertEqual(result['native_written_milestones_seconds'],[20,60])
            self.assertEqual(result['censored_native_started_at'],[71])

    def test_demo_milestones_deduplicate_starts_and_expose_unknown_timing(self):
        rows=[dict(instance_id=i,success=True,fingerprint=f,started_at=12,finished_at=end)
              for i,f,end in [('a','poseA',30),('b','poseA',40),('c','poseB',50)]]
        result=member_timing({'first_native_started_at':10},rows)
        self.assertEqual(result['time_to_distinct_demo_members_seconds'],[20,40])
        self.assertEqual(result['second_member_seconds'],40)
        self.assertIsNone(result['third_member_seconds'])
        self.assertIsNone(member_timing({'first_native_started_at':10},rows,camera=True)['second_member_seconds'])
        rows.append(dict(instance_id='old',success=True,fingerprint='poseC',started_at=None,finished_at=None))
        result=member_timing({'first_native_started_at':10},rows)
        self.assertIsNone(result['first_member_seconds'])
        self.assertEqual(result['successful_member_timing_unknown_ids'],['old'])

    def test_shared_allocation_counted_once(self):
        job={'backend':'slurm','job_id':'1','terminal':True,'allocated_gpu_seconds':3600}
        out=accounting([job,job]);self.assertEqual(out['observed_allocated_gpu_hours'],1);self.assertEqual(out['unique_allocations'],1)

    def test_censored_time_and_missing_vast_cost_retained(self):
        out=accounting([{'backend':'slurm','job_id':'1','terminal':False,'allocated_gpu_seconds':1800},{'backend':'vast','job_id':'900000001','terminal':True,'allocated_gpu_seconds':None}]);self.assertEqual(out['unfinished_consumed_gpu_hours'],.5);self.assertFalse(out['cost_complete']);self.assertIsNone(rates(out,{'demo_successful_instances':2})['gpu_hours_per_demo_successful_instances'])

    def test_eval_rate_includes_sibling_and_failed_cost(self):
        cost=accounting([{'backend':'slurm','job_id':'1','terminal':True,'allocated_gpu_seconds':3600},{'backend':'slurm','job_id':'2','terminal':True,'allocated_gpu_seconds':1800}]);out=rates(cost,{'accepted_eval_instances':1,'demo_successful_instances':3});self.assertEqual(out['gpu_hours_per_accepted_eval_instances'],1.5);self.assertEqual(out['gpu_hours_per_demo_successful_instances'],.5)

    def test_regression_pass_cannot_infer_benchmark_acceptance(self):
        cost=accounting([{'backend':'slurm','job_id':'1','terminal':True,'allocated_gpu_seconds':3600}]);out=rates(cost,{'validation_passed_instances':3,'accepted_eval_instances':0});self.assertIsNone(out['gpu_hours_per_accepted_eval_instances'])

    def test_partial_append_is_censored_not_corrupt_history(self):
        with tempfile.TemporaryDirectory()as t:
            p=Path(t)/'events';p.write_text('{"a":1}\n{"a":');self.assertEqual(json_lines(p),[{'a':1}]);p.write_text('{bad}\n')
            with self.assertRaises(ValueError):json_lines(p)

    def test_low_sample_distribution_is_explicit(self):
        self.assertEqual(sample_distribution([9])['n'],1);self.assertEqual(sample_distribution([9,3])['median'],6);self.assertIsNone(sample_distribution([])['median'])

    def vast_fixture(self, root):
        job={'job_id':'900001','job_dir':str(root/'out/source')}
        def put(path, value):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return reference(path)
        aid='900001_1_abc';begin={'kind':'assignment_begin','identity':aid,'job_id':'900001','instance':'42','assignment_started_unix':10,'scope':{'job_dir':job['job_dir']}}
        begin_ref=put(root/'vast/state/cost_provenance'/('assignment_begin_'+hashlib.sha256(('assignment_begin\0'+aid).encode()).hexdigest()+'.json'),begin)
        done={'id':'900001','attempt_id':aid,'job_dir':job['job_dir'],'gpu_count':1,'fetch_started_unix':12,'runtime_started_unix':15,'runtime_ended_unix':35,'upload_finished_unix':40,'rc':1}
        attempt=root/'vast/state/attempts/900001'/(''+aid+'.json');put(attempt,{'schema':'isg-vast-attempt-v1','job_id':'900001','attempt_id':aid,'instance':'42','assigned_unix':10,'done':done})
        state=root/'vast/state/jobs/900001.json';put(state,{'id':'900001','job_dir':job['job_dir'],'status':'failed','tries':1,'cost_assignment_begin':begin_ref,'instance':'42','assigned':10})
        return job,state,attempt,put

    def test_vast_failed_attempt_includes_fetch_and_upload_once(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);job,state,attempt,put=self.vast_fixture(root);out=vast_accounting(root,job,root/'evidence',now=100)
            self.assertEqual(out['allocated_gpu_seconds'],30);self.assertEqual(out['attempts'][0]['runtime_seconds'],20);self.assertTrue(out['terminal'])
            self.assertEqual(accounting([out,out])['observed_allocated_gpu_hours'],30/3600)

    def test_vast_foreign_attempt_fails_identity(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);job,state,attempt,put=self.vast_fixture(root);v=json.loads(attempt.read_text());v['done']['job_dir']=str(root/'foreign');put(attempt,v)
            with self.assertRaisesRegex(ValueError,'attempt identity'):vast_accounting(root,job,root/'evidence',now=100)

    def test_vast_lost_history_cannot_disappear_from_cost(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);job,state,attempt,put=self.vast_fixture(root);v=json.loads(state.read_text());v['tries']=2;put(state,v)
            out=vast_accounting(root,job,root/'evidence',now=100);self.assertIsNone(out['allocated_gpu_seconds']);self.assertEqual(out['known_attempt_gpu_seconds'],30)

    def test_vast_live_cost_requires_fresh_exact_heartbeat(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);job,state,attempt,put=self.vast_fixture(root);attempt.unlink();v=json.loads(state.read_text());v['status']='assigned';put(state,v)
            put(root/'vast/worker_heartbeats/42.json',{'heartbeat':{'time':30,'running':{'900001':{'phase':'run','cpus':'1-9'}}}})
            out=vast_accounting(root,job,root/'evidence',now=100);self.assertEqual(out['allocated_gpu_seconds'],20);self.assertFalse(out['terminal'])
            stale=vast_accounting(root,job,root/'evidence',now=1000);self.assertIsNone(stale['allocated_gpu_seconds'])

    def test_vast_timestamp_and_assignment_mismatch_rejected(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);job,state,attempt,put=self.vast_fixture(root);v=json.loads(attempt.read_text());v['done']['upload_finished_unix']=20;put(attempt,v)
            with self.assertRaisesRegex(ValueError,'timestamps'):vast_accounting(root,job,root/'evidence',now=100)



    def test_vast_replay_cost_requires_hash_bound_canonical_transport(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);job,state,attempt,put=self.vast_fixture(root)
            # The queue/attempt identities name the transported worker directory;
            # the suite owns a canonical symlink to that transport's gate.
            transport=root/'out/vast_replay/exact';canonical=root/'validation/member'
            manifest={'schema':'isg-vast-physics-replay-inputs-v1','campaign_root':str(root),
                      'job_dir':str(transport),'source_job':str(root/'out/source'),
                      'canonical_output':str(canonical)}
            ref=put(transport/'replay_inputs.json',manifest)
            canonical.parent.mkdir();canonical.symlink_to(transport/'gate')
            v=json.loads(state.read_text());v.update(job_dir=str(transport),job_kind='physics_replay',
                source_job=manifest['source_job'],replay_inputs_sha256=ref['sha256']);put(state,v)
            a=json.loads(attempt.read_text());a['done']['job_dir']=str(transport);put(attempt,a)
            bp=Path(v['cost_assignment_begin']['path']);b=json.loads(bp.read_text());b['scope']['job_dir']=str(transport);put(bp,b)
            measured=vast_accounting(root,{**job,'job_dir':str(canonical),'stage':'physics_replay'},root/'evidence',now=100)
            self.assertEqual(measured['allocated_gpu_seconds'],30)
            self.assertIn(ref,measured['evidence'])
            manifest['canonical_output']=str(root/'validation/another');put(transport/'replay_inputs.json',manifest)
            with self.assertRaisesRegex(ValueError,'manifest hash'):
                vast_accounting(root,{**job,'job_dir':str(canonical),'stage':'physics_replay'},root/'evidence',now=100)
            v['replay_inputs_sha256']=reference(transport/'replay_inputs.json')['sha256'];put(state,v)
            with self.assertRaisesRegex(ValueError,'canonical output'):
                vast_accounting(root,{**job,'job_dir':str(canonical),'stage':'physics_replay'},root/'evidence',now=100)

if __name__=='__main__':unittest.main()
