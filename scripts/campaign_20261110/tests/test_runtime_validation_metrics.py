import json
from pathlib import Path
import tempfile
import unittest
from scripts.campaign_20261110.runtime_validation_metrics import attempts, accounting, rates, register, reference, validate_row, json_lines, sample_distribution


class MetricsTest(unittest.TestCase):
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


if __name__=='__main__':unittest.main()
