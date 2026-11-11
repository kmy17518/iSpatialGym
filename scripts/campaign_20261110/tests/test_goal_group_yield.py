import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from scripts.campaign_20261110.contracts import file_digest
from scripts.campaign_20261110.goal_group_yield import summarize, member_start, snapshot, held_activity, Evidence


def group(n, status='completed', accepted=None, jobs=('1',), exact=False):
    return dict(status=status, distinct_demo_successes=n, attempted=status=='completed',
        accepted=accepted or {}, rejections={}, demo_rejections={}, acceptance_reasons=[],
        acceptance_outcome='eval_with_training_siblings' if (accepted or {}).get('eval_standard') else 'no_distinct_successful_demo',
        observed_attempted_slots=3, observed_native_draws=7, attempt_denominator_complete=exact, job_ids=list(jobs))


class GoalYieldTest(unittest.TestCase):
    def test_completion_and_histogram(self):
        rows=[group(n) for n in range(4)]+[group(0,'pending'),group(1,'running_or_collecting')]
        result=summarize(rows,[])
        self.assertEqual(result['completed_groups'],4)
        self.assertEqual(result['success_histogram'],dict.fromkeys('0123',1))
        self.assertEqual(result['at_least_two_percent'],50)
        self.assertIsNone(result['attempted_instances'])
        self.assertEqual(result['observed_attempted_slots'],12)
        self.assertIsNone(result['successful_members_per_attempted_instance'])

    def test_shared_job_cost_once_includes_zero_success_failed_job(self):
        jobs=[dict(job_id='1',terminal=True,backend='slurm',allocated_gpu_seconds=3600,activated_groups=['a','b']),
              dict(job_id='2',terminal=True,backend='slurm',allocated_gpu_seconds=3600,activated_groups=[])]
        rows=[group(3,accepted={'eval_standard':1,'train':2},exact=True),group(0,jobs=('1','2'),exact=True)]
        result=summarize(rows,jobs)
        self.assertEqual(result['gpu']['exact_slurm_gpu_hours'],2)
        self.assertEqual(result['gpu']['all_backend_accepted_per_gpu_hour'],1.5)
        self.assertEqual(result['gpu']['setup_failed_jobs_with_no_activated_group'],1)
        self.assertEqual(result['successful_members_per_attempted_instance'],.5)
        self.assertIsNone(result['gpu']['end_to_end_accepted_per_gpu_hour'])

    def test_vast_missing_startup_never_claims_all_backend_rate(self):
        result=summarize([group(3,accepted={'train':3})],[dict(job_id='1',terminal=True,backend='vast',allocated_gpu_seconds=None,activated_groups=['a'])])
        self.assertIsNone(result['gpu']['all_backend_accepted_per_gpu_hour'])
        self.assertEqual(result['gpu']['missing_job_ids'],['1'])

    def test_distinct_start_uses_acceptance_fingerprint(self):
        block={'runtime':{'goal_group':{'requested_goal':{'a':1},'requested_start':{'pose':[1,2,3]}}},
               'perturbations':{'scales':{},'lighting':{},'removed_objects':[]}}
        a=member_start(block);self.assertEqual(a,member_start(block))
        block['runtime']['goal_group']['requested_start']['grasp']='new'
        self.assertNotEqual(a,member_start(block))

    def fixture(self, root, queue=True):
        def write(path,value):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path
        cfg={'campaign_id':'campaign_20261110','split':'eval_standard','validation_key':{},
             'template':{'task':'mental_rotation','class_id':'mental_rotation-mug'}}
        plans=[]
        for gid in ('g1','g2'):
            path=write(root/'groups'/gid/'goal_plan.json',{'campaign_id':'campaign_20261110','group_id':gid,'split':'eval_standard','created_at':1})
            write(path.with_name('campaign_config.json'),cfg);plans.append(path)
        job=root/'out/job1';job.mkdir(parents=True)
        q=write(root/'queues/q.json',{'campaign_id':'campaign_20261110','config_path':str(plans[0].with_name('campaign_config.json')),
            'config_sha256':file_digest(plans[0].with_name('campaign_config.json')),
            'entries':[{'path':str(p),'sha256':file_digest(p)} for p in plans]})
        (job/'job.env').write_text(('ISPATIALGYM_CAMPAIGN_GOAL_QUEUE='+str(q)) if queue else ('ISPATIALGYM_CAMPAIGN_GOAL_PLAN='+str(plans[0])))
        write(root/'accounting/job_gpu_time.json',{'records':[{'job_id':'1','state':'COMPLETED','backend':'slurm','allocated_gpu_seconds':3600}]})
        rows=[{'jobid':'1','job_dir':str(job),'code':str(root/'code/trunk_abc'),'kcollect':'1','n':'3'}]
        return job,q,rows,write

    def test_queue_path_entries_and_unstarted_not_zero_yield(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);job,q,rows,write=self.fixture(root)
            events=[{'group_id':'g1','queue_sha256':file_digest(q),'event':'activated','created_at':2},
                    {'group_id':'g1','queue_sha256':file_digest(q),'event':'closed','created_at':3,'written':0}]
            q.with_suffix('.events.jsonl').write_text('\n'.join(json.dumps(x) for x in events)+'\n')
            with patch('scripts.campaign_20261110.goal_group_yield.snapshot_allocations',return_value=([],{},[])),patch('scripts.campaign_20261110.goal_group_yield.allocations',return_value=rows):
                result=snapshot(root)
            self.assertEqual(result['errors'],[])
            r=result['summary']['standard_eval']
            self.assertEqual(r['completed_groups'],1);self.assertEqual(r['success_histogram']['0'],1)
            self.assertEqual(r['pending_running_unstarted']['unstarted_after_job_end'],1)

    def test_single_plan_requested_three_attempted_one(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);job,q,rows,write=self.fixture(root,False)
            p=job/'inst/template/campaign_sampling_failures.jsonl';p.parent.mkdir(parents=True)
            p.write_text(json.dumps({'goal_group_id':'g1','index':100,'attempt':0,'seed':9,'created_at':2,'reason':'no_stance'})+'\n')
            with patch('scripts.campaign_20261110.goal_group_yield.snapshot_allocations',return_value=([],{},[])),patch('scripts.campaign_20261110.goal_group_yield.allocations',return_value=rows):
                result=snapshot(root)
            self.assertEqual(result['errors'],[])
            r=result['summary']['standard_eval']
            self.assertEqual(r['observed_attempted_slots'],1);self.assertIsNone(r['attempted_instances'])
            self.assertEqual(r['completed_groups'],1)

    def test_running_vast_staged_queue_uses_authored_scope_without_old_events(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);job,q,rows,write=self.fixture(root)
            rows[0]['jobid']='900001000'
            config=str(root/'groups/g1/campaign_config.json')
            (job/'job.env.before_vast_caches').write_text(
                'ISPATIALGYM_CAMPAIGN_GOAL_QUEUE='+str(q)+'\nISPATIALGYM_CAMPAIGN_CONFIG='+config)
            (job/'job.env').write_text('ISPATIALGYM_CAMPAIGN_GOAL_QUEUE='+str(job/'campaign_inputs/queue.json')+
                                      '\nISPATIALGYM_CAMPAIGN_CONFIG='+config)
            # Old authored-side closure must never be imported as executed data.
            q.with_suffix('.events.jsonl').write_text(json.dumps({'group_id':'g1','queue_sha256':file_digest(q),
                'event':'closed','created_at':3,'written':0})+'\n')
            write(root/'vast/state/jobs/900001000.json',{'status':'assigned'})
            write(root/'accounting/job_gpu_time.json',{'records':[{'job_id':'900001000','state':'assigned','backend':'vast'}]})
            with patch('scripts.campaign_20261110.goal_group_yield.snapshot_allocations',return_value=([],{},[])),patch('scripts.campaign_20261110.goal_group_yield.allocations',return_value=rows):
                result=snapshot(root)
                self.assertEqual(result['errors'],[])
                self.assertEqual(result['summary']['standard_eval']['completed_groups'],0)
                self.assertEqual(result['summary']['standard_eval']['pending_running_unstarted']['assigned_execution_unverified'],2)
                self.assertEqual(result['jobs'][0]['queue_evidence_scope'],'authored_pending_worker_sync')
                # A terminal missing executed queue remains an evidence failure.
                write(root/'vast/state/jobs/900001000.json',{'status':'synced'})
                result=snapshot(root)
                self.assertTrue(result['errors'])

    def test_unknown_execution_is_never_inferred_running(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);job,q,rows,write=self.fixture(root)
            write(root/'accounting/job_gpu_time.json',{'records':[]})
            with patch('scripts.campaign_20261110.goal_group_yield.snapshot_allocations',return_value=([],{},[])),patch('scripts.campaign_20261110.goal_group_yield.allocations',return_value=rows):
                result=snapshot(root)
            self.assertEqual(result['summary']['standard_eval']['pending_running_unstarted'],{'execution_unavailable':2})

    def test_blocked_training_queue_is_held_not_runnable_pending(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);job,q,rows,write=self.fixture(root)
            for p in (root/'groups').glob('*/campaign_config.json'):
                c=json.loads(p.read_text());c['split']='train';write(p,c)
            queue=json.loads(q.read_text());queue['config_sha256']=file_digest(root/'groups/g1/campaign_config.json')
            for e in queue['entries']:
                p=Path(e['path']);plan=json.loads(p.read_text());plan['split']='train';write(p,plan);e['sha256']=file_digest(p)
            write(q,queue)
            write(root/'accounting/job_gpu_time.json',{'records':[{'job_id':'1','state':'PENDING','backend':'slurm'}]})
            decision={'allowed':False,'gate_sha256':'verified','category':'eval_validation_hold','reason':'all100 not validated'}
            with patch('scripts.campaign_20261110.goal_group_yield.snapshot_allocations',return_value=([],{},[])),patch('scripts.campaign_20261110.goal_group_yield.allocations',return_value=rows),patch('scripts.campaign_20261110.production_gate.check_environment',return_value=decision):
                result=snapshot(root)
            self.assertEqual(result['summary']['train_only']['pending_running_unstarted'],{'held':2})
            self.assertEqual(result['jobs'][0]['dispatch_gate'],decision)
            self.assertEqual(result['summary']['train_only']['groups_attempted'],0)

    def test_held_receipt_requires_fresh_matching_authoritative_evidence(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);job,q,rows,write=self.fixture(root)
            state=write(root/'vast/state/jobs/9001.json',{'status':'assigned'})
            observation=write(root/'infra/instance.json',{'actual_status':'exited','intended_status':'stopped'})
            receipt={'schema':'campaign-job-activity-v1','campaign_id':'campaign_20261110','job_id':'9001',
                     'job_dir':str(job),'observed_at':100.,'activity':'held','verified_running':False,
                     'source_state':{'path':str(state),'sha256':file_digest(state)},
                     'authoritative_observation':{'path':str(observation),'sha256':file_digest(observation)}}
            path=write(root/'vast/activity/jobs/9001.json',receipt)
            self.assertEqual(held_activity(Evidence(root),'9001',job,state,101.)['state'],'held')
            self.assertIsNone(held_activity(Evidence(root),'9001',job,state,2000.))
            write(state,{'status':'assigned','instance':'new'})
            self.assertIsNone(held_activity(Evidence(root),'9001',job,state,101.))

    def test_duplicate_successes_and_incomplete_transfer(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);job,q,rows,write=self.fixture(root)
            q.with_suffix('.events.jsonl').write_text(json.dumps({'group_id':'g1','queue_sha256':file_digest(q),'event':'closed','created_at':3,'written':3})+'\n')
            for index in range(3):
                iid='mug__'+str(index)
                block={'goal_group_id':'g1','validation_key':{},'runtime':{'goal_group':{'requested_goal':{'a':1},'requested_start':{'a':1}}},
                       'perturbations':{'scales':{},'lighting':{},'removed_objects':[]}}
                write(job/'inst/t'/f'{iid}-tro_state.json',{'ispatialgym':{'instance_id':iid,'campaign':block}})
                write(job/'demo'/iid/'meta.json',{'instance_id':iid,'mode':'oracle','status':'success','success':True,
                    'qa':{'enabled':True},'loader':{'campaign_restoration':{'scale_verified':True}},'provenance':{'isg_root':rows[0]['code']}})
            with patch('scripts.campaign_20261110.goal_group_yield.snapshot_allocations',return_value=([],{},[])),patch('scripts.campaign_20261110.goal_group_yield.allocations',return_value=rows):
                result=snapshot(root)
                self.assertEqual(result['errors'],[])
                self.assertEqual(result['summary']['standard_eval']['success_histogram']['1'],1)
                self.assertEqual(result['groups'][0]['duplicate_successful_members'],2)
                (job/'inst/t/mug__2-tro_state.json').unlink()
                result=snapshot(root)
                self.assertEqual(result['summary']['standard_eval']['completed_groups'],0)
                self.assertEqual(result['groups'][0]['status'],'unavailable_member_evidence')

    def test_interrupted_started_attempt_still_counts_exact_denominator(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);job,q,rows,write=self.fixture(root,False)
            p=job/'inst/t/campaign_attempts.jsonl';p.parent.mkdir(parents=True)
            p.write_text(json.dumps({'event':'started','goal_group_id':'g1','attempt_id':'one','index':100,'attempt':0,'seed':9,'validation_key':{}})+'\n')
            with patch('scripts.campaign_20261110.goal_group_yield.snapshot_allocations',return_value=([],{},[])),patch('scripts.campaign_20261110.goal_group_yield.allocations',return_value=rows):result=snapshot(root)
            r=result['summary']['standard_eval'];self.assertEqual(r['attempted_instances'],1)
            self.assertEqual(r['successful_members_per_attempted_instance'],0)
            # The first live journal must remain publishable, including its
            # interrupted draw; internal sets may not leak into the artifact.
            published=json.loads(json.dumps(result))
            self.assertEqual(published['summary']['standard_eval']['observed_native_draws'],1)
            self.assertNotIn('native_attempts',published['groups'][0])

    def test_native_retries_are_separate_from_logical_member_slots(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);job,q,rows,write=self.fixture(root,False)
            p=job/'inst/t/campaign_attempts.jsonl';p.parent.mkdir(parents=True)
            records=[{'event':'started','goal_group_id':'g1','attempt_id':str(i),'index':100,
                      'attempt':i,'seed':9+i,'validation_key':{}} for i in range(3)]
            p.write_text(''.join(json.dumps(x)+'\n' for x in records))
            with patch('scripts.campaign_20261110.goal_group_yield.snapshot_allocations',return_value=([],{},[])),patch('scripts.campaign_20261110.goal_group_yield.allocations',return_value=rows):result=snapshot(root)
            r=result['summary']['standard_eval']
            self.assertEqual(r['attempted_instances'],1)
            self.assertEqual(r['native_attempted_draws'],3)
            self.assertEqual(r['successful_members_per_native_draw'],0)


if __name__=='__main__':unittest.main()
