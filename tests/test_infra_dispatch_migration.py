import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.campaign_20261110 import infra_dispatch as D
from scripts.campaign_20261110 import infra_dispatch_migration as M


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.code = self.root/'code/trunk_test'; self.code.mkdir(parents=True)
        (self.code/'.snapshot_commit').write_text('unchanged-runtime')
        D.shared_json(self.root/'manifests/family_index_reservations.json', {'reservations':{
            'old':{'start':50000000,'n':3},'new':{'start':50000100,'n':3}}})
        self.group = self.root/'groups/old'; self.group.mkdir(parents=True)
        D.shared_json(self.group/'campaign_config.json', {'validation_key': {'exact': 'same'},
            'template': {'split': 'eval_standard', 'task': 'object_scaling',
                         'class_id': 'object_scaling-apple'}})
        D.shared_json(self.group/'empty_eval_index.json', {'schema':'isg-instance-index-v1','n':0,'instances':[]})
        D.shared_json(self.group/'goal_plan.json', {'schema':'isg-pair-goal-plan-v1','group_id':'old',
            'max_members':3,'goal_pose':{'position':[1,2,3]},'created_at':12345,'budget_seconds':900})
        D.shared_json(self.group/'source.json', {'group_id':'old','source':'immutable-anchor'})
        command = ['python3',str(self.code/'scripts/trainfast/submit_fast.py'),'submit','test-key','3',
                   '--root',str(self.root),'--code',str(self.code),'--base-index','50000000','--tag','goal_groups']
        for key, value in {'ISPATIALGYM_CAMPAIGN_CONFIG':self.group/'campaign_config.json',
                           'ISPATIALGYM_CAMPAIGN_GOAL_PLAN':self.group/'goal_plan.json',
                           'EVAL_INDEX':self.group/'empty_eval_index.json','ISGF_COLLECT_EXTRA':'--no-goal-override'}.items():
            command.extend(['--extra-env',key+'='+str(value)])
        self.payload = M.read(D.enqueue(self.root,'old',command,backends=['vast_5090']))
        self.allocation = {'jobid':'123','n':'3','job_dir':str(self.root/'out/old'),'start':'50000000',
                           'key':'test-key','code':str(self.code)}
        self.receipt = self.root/'dispatch/receipts/old.json'
        D.shared_json(self.receipt, {'status':'submitted','allocation':self.allocation})

    def request(self):
        replacement = M.fresh_replacement(self.root,self.payload,'new',50000100)
        request = {'owner':'minyeong','source_payload':self.payload,'source_receipt':str(self.receipt),
                   'minimum_pending_seconds':300,'replacement':replacement,
                   'record':{'status':'submitted','allocation':self.allocation}}
        path = self.root/'dispatch/migrations/old/request.json'; D.shared_json(path,request)
        return path,request

    def test_fresh_group_changes_only_execution_identity_and_preserves_source(self):
        original = (self.group/'goal_plan.json').read_bytes()
        replacement = M.fresh_replacement(self.root,self.payload,'new',50000100)
        new = Path(replacement['directory'])
        self.assertEqual((new/'campaign_config.json').read_bytes(), (self.group/'campaign_config.json').read_bytes())
        expected = json.loads(original); expected['group_id']='new'
        self.assertEqual(M.read(new/'goal_plan.json'),expected)
        self.assertEqual((self.group/'goal_plan.json').read_bytes(),original)
        self.assertEqual(D.option(replacement['command'],'--code'),str(self.code))
        self.assertEqual(D.option(replacement['command'],'--tag'),'goal_groups')
        self.assertEqual(D.option(replacement['command'],'--base-index'),'50000100')
        self.assertEqual(M.fresh_replacement(self.root,self.payload,'new',50000100),replacement)

    def test_changed_source_fails_before_replacement(self):
        (self.group/'campaign_config.json').write_text('{}')
        with self.assertRaises(ValueError): M.fresh_replacement(self.root,self.payload,'new',50000100)
        self.assertFalse((self.root/'groups/dispatch_migrations/new').exists())

    def suite_payload(self):
        payload = copy.deepcopy(self.payload)
        payload['command'] += ['--extra-env','ISPATIALGYM_RUNTIME_SUITE_RUN=exact-suite']
        payload['identity'] = D.input_identity(self.root,payload['command'])
        return payload

    def test_suite_keeps_range_goal_paths_and_behavior_but_isolates_outputs(self):
        payload = self.suite_payload()
        with patch('scripts.campaign_20261110.runtime_suite.authorize',return_value={'row':'exact'}) as authorize:
            replacement = M.fresh_replacement(self.root,payload,'new',50000000)
        self.assertGreaterEqual(authorize.call_count,2)
        old, new = M.environment(payload['command']),M.environment(replacement['command'])
        for key in ('ISPATIALGYM_CAMPAIGN_CONFIG','ISPATIALGYM_CAMPAIGN_GOAL_PLAN','EVAL_INDEX'):
            self.assertEqual(new[key],old[key])
        self.assertEqual(D.option(replacement['command'],'--base-index'),'50000000')
        self.assertEqual(D.option(replacement['command'],'--tag'),'new')
        self.assertEqual(M.read(self.group/'goal_plan.json')['group_id'],'old')
        self.assertFalse((Path(replacement['directory'])/'goal_plan.json').exists())
        with patch('scripts.campaign_20261110.runtime_suite.authorize',return_value={'row':'exact'}):
            self.assertEqual(M.fresh_replacement(self.root,payload,'new',50000000),replacement)
            with self.assertRaisesRegex(ValueError,'different range'):
                M.fresh_replacement(self.root,payload,'different',50000100)
            changed=copy.deepcopy(replacement);changed['command'] += ['--sample-limit-s','1']
            with self.assertRaisesRegex(ValueError,'non-routing'):
                M.check_suite_replacement(self.root,payload,changed)

    def test_suite_invalid_replacement_cannot_cancel_or_publish(self):
        path, request = self.request()
        payload=self.suite_payload();request['source_payload']=payload
        D.shared_json(path,request)
        with patch.object(M.getpass,'getuser',return_value='minyeong'), \
             patch('scripts.campaign_20261110.runtime_suite.authorize',return_value={'row':'exact'}), \
             patch.object(M,'migrate_record') as cancel:
            self.assertEqual(M.cancel(self.root),[])
        cancel.assert_not_called()
        request['record']['migration']={'never_started':True,'original_allocation':self.allocation,
            'after':{'accounting':{'state':'CANCELLED','start':'Unknown'}},
            'final_execution_proof':{'nonempty_execution_artifacts':[]}}
        D.shared_json(path,request)
        with patch.object(M.getpass,'getuser',return_value='minyeong'), \
             patch('scripts.campaign_20261110.runtime_suite.authorize',return_value={'row':'exact'}), \
             patch.object(M,'enqueue') as enqueue:
            self.assertEqual(M.publish(self.root),[])
        enqueue.assert_not_called()

    def test_suite_publication_retains_exact_range_only_after_never_started_proof(self):
        payload=self.suite_payload();D.shared_json(self.root/'dispatch/ready/old.json',payload)
        with patch('scripts.campaign_20261110.runtime_suite.authorize',return_value={'row':'exact'}):
            replacement=M.fresh_replacement(self.root,payload,'new',50000000)
        request={'owner':'minyeong','source_payload':payload,'source_receipt':str(self.receipt),
                 'replacement':replacement,'record':{'allocation':self.allocation}}
        path=self.root/'dispatch/migrations/old/request.json';D.shared_json(path,request)
        with patch.object(M.getpass,'getuser',return_value='minyeong'):
            self.assertEqual(M.publish(self.root),[])
        self.assertFalse((self.root/'dispatch/ready/new.json').exists())
        request['record']['migration']={'never_started':True,'original_allocation':self.allocation,
            'after':{'accounting':{'state':'CANCELLED','start':'Unknown'}},
            'final_execution_proof':{'nonempty_execution_artifacts':[]}}
        D.shared_json(path,request)
        with patch.object(M.getpass,'getuser',return_value='minyeong'), \
             patch('scripts.campaign_20261110.runtime_suite.authorize',return_value={'row':'exact'}):
            self.assertEqual(M.publish(self.root),['new'])
        intent=M.read(self.root/'dispatch/ready/new.json')
        self.assertEqual(intent['identity']['start'],50000000)
        self.assertEqual(M.read(self.receipt)['allocation'],self.allocation)
        self.assertTrue(M.read(self.receipt)['migration']['never_started'])

    def test_queue_and_camera_owner_remain_with_existing_controller(self):
        payload = copy.deepcopy(self.payload)
        payload['command'] += ['--extra-env','ISPATIALGYM_CAMPAIGN_GOAL_QUEUE=/unused/queue.json']
        self.assertTrue(M.excluded_controller_work(self.root,payload))
        D.shared_json(self.root/'controllers/camera_v1/state.json', {'seeds':{'lane':{'production':[
            {'dispatch_receipt':str(self.receipt)}]}}})
        self.assertTrue(M.excluded_controller_work(self.root,self.payload))
        (self.root/'controllers/camera_v1/state.json').unlink()
        D.shared_json(self.root/'controllers/families/state.json', {'stage':{'dispatch_receipt':str(self.receipt)}})
        self.assertFalse(M.excluded_controller_work(self.root,self.payload))

    def test_publish_requires_cancelled_never_started_proof(self):
        path,request = self.request()
        request['record']['migration']={'never_started':True,'after':{'accounting':{'state':'RUNNING','start':'now'}},
            'final_execution_proof':{'nonempty_execution_artifacts':[]}}
        D.shared_json(path,request)
        with patch.object(M.getpass,'getuser',return_value='minyeong'), patch.object(M,'enqueue') as enqueue:
            self.assertEqual(M.publish(self.root),[])
        enqueue.assert_not_called()
        self.assertNotIn('migration',M.read(self.receipt))

    def test_publish_retains_original_allocation_and_hashes_complete_proof(self):
        path,request = self.request()
        proof={'schema':'isg-never-started-migration-v1','never_started':True,'original_allocation':self.allocation,
               'after':{'accounting':{'state':'CANCELLED','start':'Unknown'}},
               'final_execution_proof':{'nonempty_execution_artifacts':[]}}
        request['record']['migration']=proof;D.shared_json(path,request)
        with patch.object(M.getpass,'getuser',return_value='minyeong'):
            self.assertEqual(M.publish(self.root),['new'])
            self.assertEqual(M.publish(self.root),[])
        receipt=M.read(self.receipt)
        self.assertEqual(receipt['allocation'],self.allocation)
        report=receipt['migration']['proof']
        self.assertEqual(M.read(report['path']),proof)
        self.assertEqual(M.file_digest(report['path']),report['sha256'])
        intent=M.read(self.root/'dispatch/ready/new.json')
        self.assertEqual(intent['identity']['start'],50000100)
        self.assertEqual(intent['backends'],['vast_5090'])
        self.assertEqual(intent['migration_routing']['source_intent']['path'], str(self.root/'dispatch/ready/old.json'))

    def test_publish_preserves_compatible_original_clusters(self):
        self.payload['backends'] = ['shiyuc_svl','minyeong_svl','vast_5090']
        D.shared_json(self.root/'dispatch/ready/old.json', self.payload)
        path,request=self.request()
        request['record']['migration']={'never_started':True,'original_allocation':self.allocation,
            'after':{'accounting':{'state':'CANCELLED','start':'Unknown'}},
            'final_execution_proof':{'nonempty_execution_artifacts':[]}}
        D.shared_json(path,request)
        with patch.object(M.getpass,'getuser',return_value='minyeong'):
            self.assertEqual(M.publish(self.root), ['new'])
        new=M.read(self.root/'dispatch/ready/new.json')
        self.assertEqual(set(new['backends']),set(self.payload['backends']))
        self.assertNotIn('shiyuc_viscam',new['backends'])

    def test_started_during_handoff_is_not_replaced_and_unpauses_source(self):
        path,request=self.request()
        receipt=M.read(self.receipt);receipt['migration_pending']={'request':str(path)};D.shared_json(self.receipt,receipt)
        with patch.object(M.getpass,'getuser',return_value='minyeong'), patch.object(M,'migrate_record',return_value=None), \
             patch.object(M,'scheduler_proof',return_value={'scheduler':{'JobState':'RUNNING'}}):
            M.cancel(self.root)
            M.publish(self.root)
        self.assertNotIn('migration_pending',M.read(self.receipt))
        self.assertIn('migration_aborted',M.read(self.receipt))
        self.assertFalse((self.root/'dispatch/ready/new.json').exists())


if __name__ == '__main__': unittest.main()
